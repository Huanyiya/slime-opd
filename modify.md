方案可以直接落到现有 SGLang OPD 路径上，并保持 `sampled` 路径不变。

先明确一个关键点：按 Relax/SGLang 的架构，student Top-K 的 token IDs 必须在 rollout 时产生，因为 teacher 要在 actor update 前对这组 IDs 打分。actor update 时重新计算的是当前 student 在这些 IDs 上的 logprob 和最终 loss，不保存 rollout 阶段的 student Top-K 概率。

整体数据流：

```text
student SGLang rollout
  └─ output_top_logprobs → student_topk_ids [R,K]
                              │
                              ▼
teacher SGLang scoring
  └─ token_ids_logprob=flatten(student_topk_ids)+dummy_ids [R+1,K]
  └─ 丢弃 teacher dummy 行 → opd_topk_teacher_log_probs [R,K]
                              │
                              ▼
actor update 重新 forward student
  └─ 完整词表归一化后的 student logprob
  └─ gather student_topk_ids → stu_logp_topk [R_local,K]
  └─ weights = softmax(stu_logp_topk, dim=-1)
  └─ token_loss = sum(weights * (stu_logp_topk-tea_logp_topk), dim=-1)
  └─ 独立 opd_topk_loss_function 按 loss_mask 聚合并反向传播
```

Pure Top-K OPD 不进入 `policy_loss_function()`。它不计算 PPO ratio、clipped policy gradient、entropy、TIS/OIS 或 reference KL，也不要求 batch 中存在 `advantages`。

## 1. 参数设计

在 [arguments.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/utils/arguments.py:1125) 新增：

```text
--opd-loss-type sampled|topk
--opd-top-k K
```

建议：

```python
choices=["sampled", "topk"]
default="sampled"

parser.add_argument(
    "--opd-top-k",
    type=int,
    default=16,
)
```

K 不在 OPD 模块中硬编码。所有 student rollout、teacher 查询、dummy IDs、shape 校验和 actor loss 都统一使用 `args.opd_top_k`。`default=16` 只是默认配置，不是实现常量。

现有全局 `--loss-type policy_loss` 不改，因为它负责选择 PPO/SFT/custom loss，不适合作为 OPD 内部分支参数。

参数校验：

- `sampled`：维持现有 SGLang/Megatron 行为。
- `topk`：当前只允许 `--opd-type sglang`。
- `topk`：要求 `args.opd_top_k > 0`。
- 不启用 OPD 时，这个参数不起作用。

脚本 [run-qwen3-8B-opd.sh](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/examples/on_policy_distillation/run-qwen3-8B-opd.sh:154) 增加：

```bash
OPD_TOP_K=16

--opd-loss-type topk
--opd-top-k "${OPD_TOP_K}"
```

脚本用同一个 `OPD_TOP_K` shell 变量同时配置 Slime 参数和 teacher SGLang 环境变量，避免两端 K 不一致。

你安装的 SGLang 补丁实际读取的环境变量是：

```bash
RELAX_OPD_TOKEN_IDS_LOGPROB_K="${OPD_TOP_K}"
```

因此启动 teacher server 时需要设置：

```bash
RELAX_OPD_TOKEN_IDS_LOGPROB_K="${OPD_TOP_K}" \
CUDA_VISIBLE_DEVICES=6,7 "${SLIME_PYTHON}" -m sglang.launch_server ...
```

它不是 slime 命令行参数。之前提到的 `RELAX_OPD_PER_POS_TOKEN_IDS=1` 与当前安装的补丁代码不匹配。

## 2. Student rollout 获取 Top-K IDs

修改 [sglang_rollout.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/rollout/sglang_rollout.py:174)。

仅当：

```python
args.use_opd and args.opd_loss_type == "topk"
```

时，在 student SGLang 请求顶层加入：

```python
payload["top_logprobs_num"] = args.opd_top_k
```

注意它是请求顶层字段，不在 `sampling_params` 里面。

从：

```python
output["meta_info"]["output_top_logprobs"]
```

取每个 response 位置的 K 个 token ID，形成：

```text
student_topk_ids: [response_length, args.opd_top_k]
```

只保存 IDs，不保存 student rollout Top-K logprob。

建议给 `Sample` 增加独立字段：

```python
opd_topk_token_ids: Tensor | list | None
opd_topk_teacher_log_probs: Tensor | list | None
```

位置在 [types.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/utils/types.py:120)。

不复用 `teacher_log_probs` 的原因是：

- sampled 是 `[R]`
- topk 是 `[R,K]`
- 独立字段能避免已有 sampled 日志、CP 切分和 advantage 逻辑被二维张量干扰。

同时检查：

```python
student_topk_ids.shape == (sample.response_length, args.opd_top_k)
len(output_top_logprobs) == len(output_ids)
sample.response_length == len(output_ids)

```

## 3. Teacher 逐位置评分

修改 [on_policy_distillation.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/rollout/on_policy_distillation.py:140)。

### sampled 分支

完全保留现有请求和解析：

```python
logprob_start_len = 0
input_token_logprobs
teacher_log_probs: [R]
```

当前 sampled token 本身就是 student rollout 生成的 token，不是 teacher sampled token。

### topk 分支

请求 teacher 时：

```python

teacher_input_ids = sample.tokens

prompt_length = len(sample.tokens) - sample.response_length
payload["input_ids"] = teacher_input_ids
flattened_ids = student_topk_ids.reshape(-1).tolist()

payload["logprob_start_len"] = max(prompt_length - 1, 0)
payload["token_ids_logprob"] = flattened_ids + [0] * args.opd_top_k
```

长度必须满足：

```text
total_positions
    = total_length - (prompt_length - 1)
    = response_length + 1
```


teacher 返回：

```python
reward["meta_info"]["input_token_ids_logprobs"]
```

原始返回应为 `[R+1,K]`：前 R 行对应 student response 的 R 个预测位置，最后一行对应请求末尾的 dummy IDs。丢弃最后一行后得到：

```text
opd_topk_teacher_log_probs: [R,K]
```

并验证：

- 原始返回行数等于 `response_length + 1`；
- 丢弃 dummy 行后行数等于 `response_length`；
- 每行正好 `args.opd_top_k` 个元素；
- 前 R 行返回的 token IDs 与请求的 student Top-K IDs 一致；
- 最后一行是 K 个 token ID 0，只用于 SGLang 位置对齐，不参与 loss。

teacher 返回的是完整 teacher 词表 `log_softmax` 后在指定 IDs 上 gather 的值，不对这 K 个值重新归一化。

## 4. 数据传输

需要把两个新字段接入现有 Sample → Ray → actor 流程：

- [types.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/utils/types.py)
- [ray/rollout.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/ray/rollout.py:45)
- [megatron_utils/data.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/backends/megatron_utils/data.py:280)
- [megatron_utils/model.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/backends/megatron_utils/model.py:575)
- [megatron_utils/actor.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/backends/megatron_utils/actor.py:260)

数据类型：

```text
opd_topk_token_ids            int32/long  [R,K]
opd_topk_teacher_log_probs    float32     [R,K]
```

`slice_log_prob_with_cp()` 本身按第一维切片，能够扩展到 `[R,K]`，但应补测试确认 CP 下两者保持逐位置对齐。

为了避免显存膨胀，不应在 `_get_rollout_data()` 中把整个 rollout batch 的 `[R,K]` 数据全部搬到 GPU。建议：

- Ray/object store 中保存 CPU tensor。
- CP 切分可以先在 CPU 上完成。
- 除了构造 teacher HTTP 请求时临时 flatten，其他所有 Sample、Ray、rollout_data、microbatch 和 CP 切分阶段都必须保持 [R,K]。
- `get_batch()` 只把当前 microbatch 的 IDs 和 teacher logprob 搬到 GPU。
- microbatch backward 完成后随 batch 生命周期释放。

## 5. sampled 和 topk 在训练侧分流

现有 sampled OPD 在 [loss.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/backends/megatron_utils/loss.py:620) 中先计算：

```python
reverse_kl = student_log_probs - teacher_log_probs
advantages -= coef * reverse_kl
```

这条路径保持不变，但调用条件改成：

```python
args.use_opd and args.opd_loss_type == "sampled"
```

`topk` 不调用 `apply_opd_kl_to_advantages()`，也不进入 `policy_loss_function()`。Pure Top-K OPD 使用独立的 `opd_topk_loss_function()`，直接从本次 actor update forward 的 logits 计算可微 loss。

不能把 Top-K loss 写成：

```python
loss = policy_loss_function(...) + args.opd_kl_coef * opd_topk_loss
```

因为官方 Slime 的 `policy_loss_function()` 一进入就会计算：

- PPO ratio 和 clipped policy gradient；
- entropy 及 entropy loss；
- TIS/OIS/OPSM 等 off-policy 逻辑；
- reference KL；
- 并强制从 batch 中读取 `advantages`。

Pure Top-K OPD 不依赖这些输入。在 actor 训练前置阶段，也应根据：

```python
pure_topk_opd = args.use_opd and args.opd_loss_type == "topk"
```

跳过仅为 PPO/sampled OPD 服务的 old student sampled-logprob 计算和 advantage 构造。不需要为此再增加命令行参数，直接由 `opd_loss_type == "topk"` 决定。

这样可以保证：

- sampled 路径行为不变。
- topk 路径不会错误地把 `[R,K]` KL 塞进一维 advantage。
- topk loss 不会执行任何 PPO/entropy/TIS/reference-KL 计算。
- topk loss 不读取、不创建也不要求 `advantages`。
- topk student logprob 确实在 actor update 的 forward 中重新计算。

## 6. Top-K loss 的精确实现

在 [loss.py](/mnt/cpfs/users/zhy/opd/slime-OPD/slime/slime/backends/megatron_utils/loss.py:900) 增加一个完整、独立的 loss function，而不是在 `policy_loss_function()` 内部增加 helper：

```python
def opd_topk_loss_function(
    args,
    batch,
    logits,
    sum_of_sample_mean,
):
    ...
    return opd_topk_loss, metrics
```

该函数只读取：

```text
logits
batch["unconcat_tokens"]
batch["total_lengths"]
batch["response_lengths"]
batch["loss_masks"]
batch["opd_topk_token_ids"]
batch["opd_topk_teacher_log_probs"]
```

不读取：

```text
advantages
log_probs / old_log_probs / rollout_log_probs
ref_log_probs
returns
```

先根据 `total_lengths/response_lengths/CP` 取出 response 对应的 logits，再用同一组 `opd_topk_token_ids` 得到：

```python
stu_logp_topk  # [R_local,K]
tea_logp_topk  # [R_local,K]
```

严格按你的公式：

```python
weights = torch.softmax(stu_logp_topk, dim=-1)

opd_topk_per_token = (
    weights * (stu_logp_topk - tea_logp_topk)
).sum(dim=-1)
```

然后：

```python
opd_topk_loss = sum_of_sample_mean(opd_topk_per_token)
loss = args.opd_kl_coef * opd_topk_loss

return loss, {
    "loss": loss.detach(),
    "opd_topk_loss": opd_topk_loss.detach(),
}
```

在顶层 `loss_function()` dispatch 处优先分流：

```python
if args.use_opd and args.opd_loss_type == "topk":
    func = opd_topk_loss_function
else:
    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")
```

也可以写成提前返回形式，但必须保留 `loss_function()` 现有的 microbatch/DP/CP loss scaling、`num_tokens` 和 metrics 封装逻辑。核心要求是 Top-K 分支不得调用 `policy_loss_function()`。

这里：

- `weights` 不 detach，梯度经过 weights。
- `stu_logp_topk` 不在 Top-K 内重新归一化。
- `tea_logp_topk` 不重新归一化。
- 唯一的 Top-K softmax 是 `weights = softmax(stu_logp_topk)`。
- `loss_mask` 在 K 维求和以后，对 `[R]` 的 token loss 生效。
- pure Top-K OPD 的最终 loss 只是 `opd_kl_coef * opd_topk_loss`，不叠加 policy loss。
- Top-K loss 需要处理本 CP rank 没有 response token 的情况,某些 CP rank 可能没有任何有效 response logits，此时：opd_topk_per_token.numel() == 0.需要保证 loss 仍然与 logits 计算图相连：
## 7. 完整词表归一化和显存

语义上必须先做完整词表归一化，但没有必要真的长期构造并保存：

```python
stu_logp_all: [R,V]
```

更合适的实现是计算：

```python
stu_logp_topk =
    selected_student_logits
    - logsumexp(student_logits_over_full_vocab)
```

它与下面完全等价：

```python
stu_logp_all = log_softmax(full_vocab_logits, dim=-1)
stu_logp_topk = stu_logp_all.gather(-1, topk_ids)
```

但不会把 `[R,V]` logprob 放进 `Sample`、`rollout_data` 或训练缓存。

考虑 Megatron 的 logits 是 TP vocabulary shard，建议扩展现有 vocab-parallel logprob 算子，使 target 从 `[T]` 支持到 `[T,K]`：

- TP 间计算全词表 global max 和 global exp sum。
- 得到完整词表的 log-normalizer。
- 每个 TP rank只 gather 属于自己词表分片的 Top-K logits。
- TP all-reduce 得到 `[T,K]` 的 student logprob。
- 不 all-gather `[T,V]` 到每张 GPU。
- 按 `log_probs_chunk_size` 分块处理位置。

当前脚本是 `tensor-model-parallel-size=1`，但按这个方式实现可以同时保证以后 TP>1 时仍然正确和省显存。

## 8. 日志和测试

新增训练指标：

```text
train/opd_topk_loss
```

可以再记录一个 detach 后的：

```text
train/opd_topk_reverse_kl
```

只记录 K 维求和后的 `[R]`，不要记录或长期保存完整词表概率。

测试至少包括：

- `--opd-loss-type` 只能选 `sampled/topk`。
- `--opd-top-k` 必须是正整数，并验证非默认 K（例如 K=8）能贯通整条路径。
- `sampled` 默认值和旧行为不变。
- student `output_top_logprobs → [R,K] IDs` 解析。
- teacher payload 的 `logprob_start_len` 和扁平 ID 长度正确。
- teacher 请求尾部添加 K 个 dummy zero IDs，原始返回为 `[R+1,K]` 并丢弃最后一行。
- teacher 返回 `[R,K]` 与 IDs 对齐。
- toy logits 对比显式 `log_softmax(...).gather(...)`。
- 验证只对 `weights` 做 Top-K softmax。
- 验证 weights 没有 detach，student logits 有梯度。
- sampled OPD regression。
- CP 对二维 `[R,K]` 的切分测试。
- 最后用真实 patched SGLang 做一个短 response smoke test。

另外，我看到当前 slime worktree 已经有不少未提交修改，其中包含这次会涉及的文件。实际实施时需要在现有改动上做增量 patch，不能覆盖这些用户修改。
