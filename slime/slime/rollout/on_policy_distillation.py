import json
import logging
import os
import re
from pathlib import Path

import aiohttp
import numpy as np
import pybase64
import torch
import wandb

from slime.rollout.rm_hub.math_utils import (
    extract_boxed_answer,
    grade_answer_mathd,
    grade_answer_sympy,
)
from slime.utils import logging_utils
from slime.utils.metric_utils import add_train_step_metric, compute_pass_rate, get_metric_train_step
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


logger = logging.getLogger(__name__)

_EVAL_ROLLOUT_STREAM_HANDLE = None
_EVAL_ROLLOUT_STREAM_PATH: Path | None = None


SPECIAL_RESPONSE_TOKENS = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|eot_id|>",
)

ANSWER_PROMPT_RE = re.compile(
    r"\s*Solve the following math problem step by step\.\s*"
    r"The last line of your response should be of the form Answer: \$Answer "
    r"\(without quotes\) where \$Answer is the answer to the problem\.\s*"
    r"Remember to put your answer on its own line after \"Answer:\"\.?",
    flags=re.IGNORECASE,
)

BOXED_PROMPT_INSTRUCTION = r" Please reason step by step, and put your final answer within \boxed{}."


def _define_eval_step_metrics(args, metric_names: list[str]) -> None:
    if not getattr(args, "use_wandb", False):
        return
    for metric_name in metric_names:
        wandb.define_metric(metric_name, step_metric="step")


def _strip_special_response_tokens(response: str) -> str:
    for token in SPECIAL_RESPONSE_TOKENS:
        response = response.replace(token, "")
    return response.strip()


def rewrite_answer_prompt_to_boxed(prompt: str) -> str:
    """Rewrite old Answer: eval prompts to match processed boxed training prompts."""
    return ANSWER_PROMPT_RE.sub(lambda _: BOXED_PROMPT_INSTRUCTION, prompt)


async def boxed_prompt_generate_func(args, sample: Sample, sampling_params, evaluation: bool = False):
    """Generate after normalizing old Answer: prompts to boxed-answer prompts.

    The OPD evaluation protocol asks for final answers within ``\\boxed{}``.
    Some eval sets still ask for ``Answer:``; this keeps eval prompt style aligned
    with training without editing the parquet files.
    """
    if isinstance(sample.prompt, str):
        sample.prompt = rewrite_answer_prompt_to_boxed(sample.prompt)

    from slime.rollout.sglang_rollout import generate

    return await generate(args, sample, sampling_params, evaluation=evaluation)


def log_eval_metrics(rollout_id, args, data, extra_metrics=None):
    """Log only pass@1/4/16 and mean@16 for each OPD eval dataset."""
    del extra_metrics

    group_size = args.n_samples_per_eval_prompt
    if group_size != 16:
        raise ValueError(
            "OPD eval logging requires --n-samples-per-eval-prompt 16 "
            f"to report pass@16 and mean@16, got {group_size}."
        )

    log_dict = {}
    for dataset_name, dataset_result in data.items():
        rewards = dataset_result["rewards"]
        pass_rates = compute_pass_rate(flat_rewards=rewards, group_size=group_size)
        metric_prefix = f"eval/{dataset_name}/"
        for k in (1, 4, 16):
            log_dict[f"{metric_prefix}pass@{k}"] = pass_rates[f"pass@{k}"]
        log_dict[f"{metric_prefix}mean@16"] = sum(rewards) / len(rewards)

    step = get_metric_train_step(args, fallback=0)
    add_train_step_metric(log_dict, step)
    _define_eval_step_metrics(args, [key for key in log_dict if key.startswith("eval/")])
    logger.info("eval %s: %s", rollout_id, log_dict)
    logging_utils.log(args, log_dict, step_key="step")
    return True


def log_eval_metrics_64(rollout_id, args, data, extra_metrics=None):
    """Log and persist pass@1 through pass@64 for 64 samples per evaluation prompt."""
    del extra_metrics

    group_size = args.n_samples_per_eval_prompt
    if group_size != 64:
        raise ValueError(f"pass@64 evaluation requires 64 samples per prompt, got {group_size}.")

    results = {}
    log_dict = {}
    for dataset_name, dataset_result in data.items():
        rewards = dataset_result["rewards"]
        if len(rewards) % group_size != 0:
            raise ValueError(
                f"{dataset_name} returned {len(rewards)} rewards, which is not divisible by {group_size}."
            )
        pass_rates = compute_pass_rate(flat_rewards=rewards, group_size=group_size)
        dataset_metrics = {
            metric_name: float(metric_value) for metric_name, metric_value in pass_rates.items()
        } | {
            "mean@64": float(sum(rewards) / len(rewards)),
            "num_prompts": len(rewards) // group_size,
            "num_generations": len(rewards),
        }
        results[dataset_name] = dataset_metrics
        for metric_name, metric_value in pass_rates.items():
            log_dict[f"eval/{dataset_name}/{metric_name}"] = float(metric_value)
        log_dict[f"eval/{dataset_name}/mean@64"] = dataset_metrics["mean@64"]

    model_path = os.environ.get("OPD_EVAL_MODEL_PATH")
    if model_path:
        results["model_path"] = model_path

    output_path = Path(os.environ.get("OPD_EVAL_RESULT_PATH", "eval_iter_0000179_pass64.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rollout_output_path = os.environ.get("OPD_EVAL_ROLLOUT_PATH")
    if rollout_output_path:
        rollout_output_path = Path(rollout_output_path)
        _write_eval_rollouts(data, rollout_output_path, group_size=group_size)
        logger.info("saved per-rollout evaluation results to %s", rollout_output_path)

    step = get_metric_train_step(args, fallback=0)
    add_train_step_metric(log_dict, step)
    _define_eval_step_metrics(args, [key for key in log_dict if key.startswith("eval/")])
    logger.info("eval %s 64-sample results: %s", rollout_id, json.dumps(results, ensure_ascii=False))
    logger.info("saved 64-sample results to %s", output_path)
    logging_utils.log(args, log_dict, step_key="step")
    return True


def _build_eval_rollout_record(
    dataset_name: str,
    sample: Sample,
    reward,
    *,
    position: int,
    group_size: int,
    is_truncated: bool,
) -> dict:
    response = _strip_special_response_tokens(sample.response)
    status = sample.status.value if isinstance(sample.status, Sample.Status) else str(sample.status)
    return {
        "dataset": dataset_name,
        "prompt_index": position // group_size,
        "rollout_index": position % group_size,
        "sample_index": sample.index,
        "prompt": sample.prompt,
        "response": response,
        "label": sample.label,
        "reward": reward,
        "correct": bool(reward),
        "response_token_length": sample.response_length,
        "status": status,
        "truncated": bool(is_truncated),
        "boxed_answer": extract_boxed_answer(response),
    }


def _reset_eval_rollout_stream() -> None:
    """Create an empty per-rollout JSONL file at the start of an evaluation."""
    global _EVAL_ROLLOUT_STREAM_HANDLE, _EVAL_ROLLOUT_STREAM_PATH

    _close_eval_rollout_stream()
    output_path = os.environ.get("OPD_EVAL_ROLLOUT_PATH")
    if not output_path:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _EVAL_ROLLOUT_STREAM_HANDLE = output_path.open("w", encoding="utf-8")
    _EVAL_ROLLOUT_STREAM_PATH = output_path
    logger.info("streaming per-rollout evaluation results to %s", output_path)


def _close_eval_rollout_stream() -> None:
    """Flush and close the per-rollout stream handle, if one is open."""
    global _EVAL_ROLLOUT_STREAM_HANDLE, _EVAL_ROLLOUT_STREAM_PATH

    if _EVAL_ROLLOUT_STREAM_HANDLE is not None:
        _EVAL_ROLLOUT_STREAM_HANDLE.flush()
        _EVAL_ROLLOUT_STREAM_HANDLE.close()
    _EVAL_ROLLOUT_STREAM_HANDLE = None
    _EVAL_ROLLOUT_STREAM_PATH = None


def _append_eval_rollout_stream(dataset_name: str, sample: Sample, *, group_size: int) -> None:
    """Append one completed and rewarded evaluation rollout to the JSONL file."""
    global _EVAL_ROLLOUT_STREAM_HANDLE, _EVAL_ROLLOUT_STREAM_PATH

    output_path = os.environ.get("OPD_EVAL_ROLLOUT_PATH")
    if not output_path:
        return
    output_path = Path(output_path)
    if _EVAL_ROLLOUT_STREAM_HANDLE is None or _EVAL_ROLLOUT_STREAM_PATH != output_path:
        _close_eval_rollout_stream()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _EVAL_ROLLOUT_STREAM_HANDLE = output_path.open("a", encoding="utf-8")
        _EVAL_ROLLOUT_STREAM_PATH = output_path
    position = int(sample.index)
    record = _build_eval_rollout_record(
        dataset_name,
        sample,
        sample.reward,
        position=position,
        group_size=group_size,
        is_truncated=sample.status == Sample.Status.TRUNCATED,
    )
    # Evaluation coroutines share one asyncio event loop, so this write has no
    # concurrent await point. Keep the handle open to avoid open/close overhead,
    # but flush each completed record so progress is immediately observable.
    _EVAL_ROLLOUT_STREAM_HANDLE.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    _EVAL_ROLLOUT_STREAM_HANDLE.flush()


def _write_eval_rollouts(data, output_path: Path, *, group_size: int) -> None:
    """Rewrite completed evaluation rollouts in deterministic prompt order."""
    _close_eval_rollout_stream()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for dataset_name, dataset_result in data.items():
            samples = dataset_result["samples"]
            rewards = dataset_result["rewards"]
            truncated = dataset_result["truncated"]
            if not (len(samples) == len(rewards) == len(truncated)):
                raise ValueError(
                    f"{dataset_name} per-rollout fields must have the same length: "
                    f"samples={len(samples)}, rewards={len(rewards)}, truncated={len(truncated)}."
                )

            for position, (sample, reward, is_truncated) in enumerate(
                zip(samples, rewards, truncated, strict=True)
            ):
                record = _build_eval_rollout_record(
                    dataset_name,
                    sample,
                    reward,
                    position=position,
                    group_size=group_size,
                    is_truncated=is_truncated,
                )
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _use_topk_opd(args) -> bool:
    return bool(
        getattr(args, "use_opd", False) and getattr(args, "opd_loss_type", "sampled") in {"topk", "topk_detatch"}
    )


def _decode_base64_array(meta_info: dict, field_name: str, dtype, expected_size: int) -> np.ndarray:
    encoded = meta_info.get(field_name)
    if not isinstance(encoded, str):
        raise ValueError(f"Teacher response is missing base64 field {field_name!r}.")

    try:
        raw = pybase64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Teacher response field {field_name!r} is not valid base64.") from exc

    np_dtype = np.dtype(dtype)
    expected_bytes = expected_size * np_dtype.itemsize
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Teacher response field {field_name!r} has {len(raw)} decoded bytes, "
            f"expected {expected_bytes} for {expected_size} {np_dtype.name} values."
        )
    return np.frombuffer(raw, dtype=np_dtype, count=expected_size)


def _describe_row_lengths(row_lengths) -> str:
    """Describe response structure without dumping per-token data to logs."""
    if row_lengths is None:
        return "missing"
    if not isinstance(row_lengths, list):
        return f"type={type(row_lengths).__name__}"
    unique = sorted(set(row_lengths)) if row_lengths else []
    return f"rows={len(row_lengths)}, unique_lengths={unique}"


def _build_teacher_payload(args, sample: Sample) -> dict:
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if _use_topk_opd(args):
        topk_ids = torch.as_tensor(sample.opd_topk_token_ids, dtype=torch.int32)
        expected_shape = (sample.response_length, args.opd_top_k)
        if tuple(topk_ids.shape) != expected_shape:
            raise ValueError(
                "opd_topk_token_ids must match [response_length, opd_top_k] before teacher scoring: "
                f"got={tuple(topk_ids.shape)}, expected={expected_shape}."
            )
        prompt_length = len(sample.tokens) - sample.response_length
        if prompt_length <= 0:
            raise ValueError(f"Top-K OPD teacher scoring requires a non-empty prompt, got length {prompt_length}.")
        payload["return_logprobs_in_base64"] = True
        payload["top_logprobs_num"] = args.opd_top_k
        payload["logprob_start_len"] = prompt_length - 1
        payload["token_ids_logprob"] = topk_ids.reshape(-1).tolist() + [0] * args.opd_top_k

    return payload


async def reward_func(args, sample, **kwargs):
    payload = _build_teacher_payload(args, sample)
    # Sequential OPD talks directly to an SGLang worker.  The Rust router in
    # sglang_router 0.3.2 drops extension request fields such as
    # return_logprobs_in_base64 and top_logprobs_num, which makes the response
    # unusable for Top-K distillation even though the worker returns HTTP 200.
    rm_url = kwargs.get("rm_url", args.rm_url)

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def math_eval_reward_func(args, sample: Sample, **kwargs):
    """Score OPD math evaluation samples using reward_model.ground_truth."""
    label = sample.label
    if isinstance(label, dict):
        label = label.get("ground_truth")
    if label is None:
        raise ValueError("Math evaluation sample is missing reward_model.ground_truth")

    ground_truth = str(label)
    response = _strip_special_response_tokens(sample.response)
    # Accept both \boxed{answer} and \boxed{{answer}}. The latter extracts an
    # extra brace pair, which the math normalizers handle correctly.
    boxed_answer = extract_boxed_answer(response)
    is_correct = boxed_answer is not None and (
        grade_answer_mathd(boxed_answer, ground_truth) or grade_answer_sympy(boxed_answer, ground_truth)
    )
    # Keep the legacy Answer: form as a lenient fallback so formatting alone
    # does not turn a mathematically correct response into a false negative.
    answer_matches = re.findall(r"(?i)Answer\s*[:：]\s*([^\n]+)", response)
    if answer_matches:
        answer = answer_matches[-1].strip()
        is_correct = (
            is_correct or grade_answer_mathd(answer, ground_truth) or grade_answer_sympy(answer, ground_truth)
        )
    return 1.0 if is_correct else 0.0


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores sampled-token or student-Top-K teacher log-probs on the Sample
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The learning signal is applied either through sampled-token advantages or the
    standalone Top-K OPD loss during the actor update.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    if _use_topk_opd(args):
        for sample, reward in zip(samples, raw_rewards, strict=True):
            requested_ids = torch.as_tensor(sample.opd_topk_token_ids, dtype=torch.int32)
            expected_shape = (sample.response_length, args.opd_top_k)
            if tuple(requested_ids.shape) != expected_shape:
                raise ValueError(
                    "opd_topk_token_ids must match [response_length, opd_top_k] during teacher post-processing: "
                    f"got={tuple(requested_ids.shape)}, expected={expected_shape}."
                )

            meta_info = reward["meta_info"]
            row_lengths = meta_info.get("input_token_ids_logprobs_len")
            # SGLang represents the first input-token logprob as an empty row:
            # there is no preceding position inside the returned span.  The K
            # requested IDs for that position are nevertheless used to gather
            # the first response-token distribution.  SGLang then drops the
            # final sampling position, whose requested dummy IDs only preserve
            # the per-position request shape.  Consequently the serialized
            # arrays contain exactly response_length * K values.
            expected_row_lengths = [0] + [args.opd_top_k] * sample.response_length
            if row_lengths != expected_row_lengths:
                raise ValueError(
                    "Teacher input_token_ids_logprobs_len must contain an empty leading row and K for every "
                    "response position: "
                    f"got={_describe_row_lengths(row_lengths)}, expected_rows={sample.response_length + 1}, "
                    f"K={args.opd_top_k}, meta_info_keys={sorted(meta_info)}, "
                    f"finish_reason={meta_info.get('finish_reason')!r}."
                )

            expected_size = sample.response_length * args.opd_top_k
            teacher_log_probs = _decode_base64_array(
                meta_info,
                "input_token_ids_logprobs_val_b64",
                np.float32,
                expected_size,
            ).reshape(sample.response_length, args.opd_top_k)
            teacher_ids = _decode_base64_array(
                meta_info,
                "input_token_ids_logprobs_idx_b64",
                np.int32,
                expected_size,
            ).reshape(sample.response_length, args.opd_top_k)

            teacher_topk_row_lengths = meta_info.get("input_top_logprobs_k")
            if teacher_topk_row_lengths != expected_row_lengths:
                raise ValueError(
                    "Teacher input_top_logprobs_k must contain an empty leading row and K for every response "
                    "position: "
                    f"got={_describe_row_lengths(teacher_topk_row_lengths)}, "
                    f"expected_rows={sample.response_length + 1}, K={args.opd_top_k}."
                )
            teacher_topk_ids = _decode_base64_array(
                meta_info,
                "input_top_logprobs_idx_b64",
                np.int32,
                expected_size,
            ).reshape(sample.response_length, args.opd_top_k)

            teacher_ids_tensor = torch.from_numpy(teacher_ids.copy())
            if not torch.equal(teacher_ids_tensor, requested_ids):
                raise ValueError("Teacher returned token IDs that do not match the requested student Top-K IDs.")
            sample.opd_topk_teacher_log_probs = torch.from_numpy(teacher_log_probs.copy())

            teacher_topk_ids_tensor = torch.from_numpy(teacher_topk_ids.copy())
            if sample.loss_mask is None:
                valid_mask = torch.ones(sample.response_length, dtype=torch.bool)
            else:
                valid_mask = torch.as_tensor(sample.loss_mask, dtype=torch.bool)
                if valid_mask.numel() != sample.response_length:
                    raise ValueError(
                        "loss_mask must align with response positions when computing Top-K overlap: "
                        f"got={valid_mask.numel()}, expected={sample.response_length}."
                    )

            valid_position_count = int(valid_mask.sum().item())
            for cutoff in sorted({1, 2, 4, args.opd_top_k}):
                if cutoff > args.opd_top_k:
                    continue
                overlap_count_per_position = (
                    requested_ids[:, :cutoff]
                    .unsqueeze(-1)
                    .eq(teacher_topk_ids_tensor[:, :cutoff].unsqueeze(-2))
                    .any(dim=-1)
                    .sum(dim=-1)
                )
                metric_prefix = "topk" if cutoff == args.opd_top_k else f"top{cutoff}"
                sample.metadata[f"{metric_prefix}_overlap_count"] = int(
                    overlap_count_per_position[valid_mask].sum().item()
                )
                sample.metadata[f"{metric_prefix}_overlap_total"] = valid_position_count * cutoff

            teacher_top1_matches = requested_ids.eq(teacher_topk_ids_tensor[:, :1])
            teacher_top1_in_student_topk = teacher_top1_matches.any(dim=-1)
            teacher_top1_student_rank = torch.where(
                teacher_top1_in_student_topk,
                teacher_top1_matches.to(torch.int64).argmax(dim=-1) + 1,
                torch.full(
                    (sample.response_length,),
                    args.opd_top_k + 1,
                    dtype=torch.int64,
                ),
            )
            sample.metadata["teacher_top1_in_student_topk_count"] = int(
                teacher_top1_in_student_topk[valid_mask].sum().item()
            )
            sample.metadata["teacher_top1_in_student_topk_total"] = valid_position_count
            sample.metadata["teacher_top1_in_student_rank_sum"] = int(
                teacher_top1_student_rank[valid_mask].sum().item()
            )
            sample.metadata["teacher_top1_in_student_rank_total"] = valid_position_count
    else:
        # Keep the original sampled-token OPD path unchanged.
        teacher_log_probs = [
            torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
            for reward in raw_rewards
        ]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]

        for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
            sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
