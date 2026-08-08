6 个时间指标：
perf/student_rollout_time
4B student 在 SGLang 中自回归生成 response 的墙钟时间。

perf/teacher_score_time
9B teacher 对 student 的完整 token 序列执行 max_new_tokens=0 的 forward，计算 token-level logprob 的墙钟时间。

perf/student_logprob_forward_time
Megatron student 在训练前重新 forward，计算当前 student logprob。对应你说的“student 第二次 forward”。

perf/student_train_time
真正训练阶段的总时间，包括：
training forward
+ backward
+ 梯度通信
+ optimizer.step
这一项不能省略。Student 第二次 forward 只是计算 logprob，不会更新参数。

perf/weight_sync_time
optimizer 更新完成后，将 Megatron student 新权重同步到 SGLang rollout engine 的时间。

perf/step_total_time
从本步 student rollout 开始，到新权重同步完成的总墙钟时间。用于检查各阶段之外还有多少调度、数据转换和通信开销。