import logging
import math
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id


def compute_train_steps_per_rollout(args) -> int:
    """Return how many optimizer steps one rollout produces."""
    if getattr(args, "num_steps_per_rollout", None) is not None:
        return int(args.num_steps_per_rollout)
    return int(args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size)


def compute_train_step_before_rollout(args, rollout_id: int) -> int:
    """0-based completed optimizer-step count before ``rollout_id`` starts."""
    return int(rollout_id) * compute_train_steps_per_rollout(args)


def compute_train_step_after_rollout(args, rollout_id: int) -> int:
    """Completed optimizer-step count after ``rollout_id`` finishes training."""
    return (int(rollout_id) + 1) * compute_train_steps_per_rollout(args)


def compute_first_train_step_for_rollout(args, rollout_id: int) -> int:
    """1-based optimizer step that first consumes data from ``rollout_id``."""
    return compute_train_step_before_rollout(args, rollout_id) + 1


def get_metric_train_step(args, fallback: int | None = None) -> int:
    """Read the currently injected metric step, falling back when absent."""
    step = getattr(args, "metric_train_step", None)
    if step is None:
        if fallback is None:
            raise ValueError("metric_train_step is not set and no fallback was provided")
        step = fallback
    return int(step)


def add_train_step_metric(log_dict: dict[str, Any], step: int) -> dict[str, Any]:
    """Add the single canonical W&B/TensorBoard x-axis key."""
    log_dict["step"] = int(step)
    return log_dict
