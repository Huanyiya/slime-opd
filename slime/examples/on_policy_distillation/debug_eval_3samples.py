#!/usr/bin/env python3
"""Generate 3 eval answers and save a human-readable debug txt.

Examples:

  # Use an already running SGLang server, closest to Slime eval generation.
  /mnt/cpfs/users/zhy/opd/slime-OPD/.venv/bin/python \
    /mnt/cpfs/users/zhy/opd/slime-OPD/slime/examples/on_policy_distillation/debug_eval_3samples.py \
    --backend sglang \
    --url http://127.0.0.1:30000/generate

  # Or load the model directly with Transformers on one visible device.
  CUDA_VISIBLE_DEVICES=0 /mnt/cpfs/users/zhy/opd/slime-OPD/.venv/bin/python \
    /mnt/cpfs/users/zhy/opd/slime-OPD/slime/examples/on_policy_distillation/debug_eval_3samples.py \
    --backend transformers
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from slime.rollout.rm_hub.math_utils import (  # noqa: E402
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
    grade_answer_verl,
)
from slime.rollout.on_policy_distillation import (  # noqa: E402
    _strip_special_response_tokens,
    rewrite_answer_prompt_to_boxed,
)


DEFAULT_DATASET = "/mnt/cpfs/users/zhy/opd/OPD/datasets/test_data/AIME25/test.parquet"
DEFAULT_MODEL = "/mnt/cpfs/weights/Qwen3.5-4B"
DEFAULT_OUTPUT = "/mnt/cpfs/users/zhy/opd/slime-OPD/eval_debug_3samples.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Eval parquet path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model path, used for tokenizer and transformers mode.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to save the txt debug report.")
    parser.add_argument("--backend", choices=("sglang", "transformers"), default="sglang")
    parser.add_argument("--url", default="http://127.0.0.1:30000/generate", help="SGLang /generate URL.")
    parser.add_argument("--input-key", default="prompt")
    parser.add_argument("--label-key", default="reward_model")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="If omitted, use enable_thinking=False, matching run-qwen3-8B-opd.sh.",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Decode/generate with skip_special_tokens=True. Slime currently defaults to False unless set otherwise.",
    )
    parser.add_argument(
        "--no-rewrite-answer-prompt-to-boxed",
        action="store_true",
        help="Disable OPD eval's Answer:-to-boxed prompt rewrite for comparison.",
    )
    parser.add_argument("--device-map", default="auto", help="Only used by --backend transformers.")
    return parser.parse_args()


def normalize_messages(raw_prompt: Any) -> list[dict[str, Any]]:
    if isinstance(raw_prompt, list):
        return raw_prompt
    if hasattr(raw_prompt, "tolist"):
        return raw_prompt.tolist()
    if isinstance(raw_prompt, str):
        stripped = raw_prompt.strip()
        if stripped.startswith("["):
            value = ast.literal_eval(stripped)
            if isinstance(value, list):
                return value
        return [{"role": "user", "content": raw_prompt}]
    raise TypeError(f"Unsupported prompt type: {type(raw_prompt)}")


def ground_truth_from_label(label: Any) -> str:
    if isinstance(label, dict):
        label = label.get("ground_truth")
    elif isinstance(label, str):
        try:
            parsed = ast.literal_eval(label)
            if isinstance(parsed, dict):
                label = parsed.get("ground_truth")
        except Exception:
            pass
    if label is None:
        return ""
    return str(label)


def current_eval_score(response: str, ground_truth: str) -> tuple[float, str | None]:
    """Match slime.rollout.on_policy_distillation.math_eval_reward_func exactly."""
    response = _strip_special_response_tokens(response)
    is_correct = grade_answer_verl(response, ground_truth)
    parsed_answer = extract_answer(response)

    answer_matches = re.findall(r"(?i)Answer\s*[:：]\s*([^\n]+)", response)
    if answer_matches:
        parsed_answer = answer_matches[-1].strip()
        is_correct = (
            is_correct
            or grade_answer_mathd(parsed_answer, ground_truth)
            or grade_answer_sympy(parsed_answer, ground_truth)
        )
    return (1.0 if is_correct else 0.0), parsed_answer


def lenient_answer_for_debug(response: str) -> str | None:
    """A debug-only guess, not used by current Slime eval."""
    patterns = [
        r"(?i)Answer\s*[:：]\s*([^\n]+)",
        r"(?i)final answer\s*[:：]\s*([^\n]+)",
        r"(?i)the answer is\s*([^\n.。]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            return matches[-1].strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", response)
    return numbers[-1] if numbers else None


def render_prompt(tokenizer: Any, messages: list[dict[str, Any]], enable_thinking: bool) -> str:
    kwargs = {} if enable_thinking else {"enable_thinking": False}
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def generate_with_sglang(args: argparse.Namespace, tokenizer: Any, rendered_prompt: str) -> dict[str, Any]:
    import requests

    prompt_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "skip_special_tokens": args.skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    response = requests.post(args.url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    data = response.json()

    meta_info = data.get("meta_info", {})
    token_logprobs = meta_info.get("output_token_logprobs") or []
    finish_reason = meta_info.get("finish_reason")
    truncated = len(token_logprobs) >= args.max_new_tokens or "length" in str(finish_reason).lower()
    return {
        "text": data.get("text", ""),
        "response_tokens": len(token_logprobs),
        "finish_reason": finish_reason,
        "truncated": truncated,
        "raw_meta_keys": sorted(meta_info.keys()),
    }


def generate_with_transformers(args: argparse.Namespace, tokenizer: Any, rendered_prompt: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    do_sample = args.temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=args.skip_special_tokens)
    return {
        "text": text,
        "response_tokens": int(new_ids.numel()),
        "finish_reason": None,
        "truncated": int(new_ids.numel()) >= args.max_new_tokens,
        "raw_meta_keys": [],
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    df = pd.read_parquet(args.dataset)
    end = min(args.start + args.num_samples, len(df))
    rows = df.iloc[args.start:end]

    generator = generate_with_sglang if args.backend == "sglang" else generate_with_transformers
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: list[str] = []
    report.append(f"created_at: {datetime.now().isoformat(timespec='seconds')}")
    report.append(f"dataset: {args.dataset}")
    report.append(f"model: {args.model}")
    report.append(f"backend: {args.backend}")
    report.append(f"url: {args.url if args.backend == 'sglang' else '(unused)'}")
    report.append(f"enable_thinking: {args.enable_thinking}")
    report.append(f"rewrite_answer_prompt_to_boxed: {not args.no_rewrite_answer_prompt_to_boxed}")
    report.append(f"max_new_tokens: {args.max_new_tokens}")
    report.append(f"temperature: {args.temperature}")
    report.append(f"top_p: {args.top_p}")
    report.append("")

    for local_i, (_, row) in enumerate(rows.iterrows(), start=1):
        messages = normalize_messages(row[args.input_key])
        ground_truth = ground_truth_from_label(row[args.label_key])
        rendered = render_prompt(tokenizer, messages, enable_thinking=args.enable_thinking)
        if not args.no_rewrite_answer_prompt_to_boxed:
            rendered = rewrite_answer_prompt_to_boxed(rendered)
        generated = generator(args, tokenizer, rendered)
        response = generated["text"]
        score, parsed_answer = current_eval_score(response, ground_truth)
        lenient_answer = lenient_answer_for_debug(response)
        lenient_score = (
            grade_answer_mathd(lenient_answer, ground_truth) or grade_answer_sympy(lenient_answer, ground_truth)
            if lenient_answer is not None
            else False
        )

        report.append("=" * 100)
        report.append(f"sample: {local_i}")
        report.append(f"dataset_row_index: {args.start + local_i - 1}")
        report.append(f"ground_truth: {ground_truth}")
        report.append(f"current_eval_score: {score}")
        report.append(f"current_eval_parsed_answer: {parsed_answer}")
        report.append(f"lenient_debug_answer: {lenient_answer}")
        report.append(f"lenient_debug_score: {float(lenient_score)}")
        report.append(f"response_tokens: {generated['response_tokens']}")
        report.append(f"truncated_or_hit_max_new_tokens: {generated['truncated']}")
        report.append(f"finish_reason: {generated['finish_reason']}")
        report.append(f"raw_meta_keys: {generated['raw_meta_keys']}")
        report.append("rendered_prompt_tail:")
        report.append(rendered[-1200:])
        report.append("prompt_messages:")
        report.append(json.dumps(messages, ensure_ascii=False, indent=2))
        report.append("response:")
        report.append(response)
        report.append("")

    output_path.write_text("\n".join(report), encoding="utf-8")
    print(f"saved debug report to: {output_path}")


if __name__ == "__main__":
    main()
