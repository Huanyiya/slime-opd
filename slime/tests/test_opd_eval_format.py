import asyncio
import json
from pathlib import Path

from slime.rollout.on_policy_distillation import (
    _append_eval_rollout_stream,
    _close_eval_rollout_stream,
    _reset_eval_rollout_stream,
    _write_eval_rollouts,
    math_eval_reward_func,
    rewrite_answer_prompt_to_boxed,
)
from slime.utils.types import Sample


def test_rewrite_answer_prompt_requests_single_boxed_wrapper():
    prompt = (
        "Compute 1+1. Solve the following math problem step by step. "
        "The last line of your response should be of the form Answer: $Answer "
        "(without quotes) where $Answer is the answer to the problem. "
        'Remember to put your answer on its own line after "Answer:".'
    )

    rewritten = rewrite_answer_prompt_to_boxed(prompt)

    assert rewritten.endswith(r"Please reason step by step, and put your final answer within \boxed{}.")
    assert r"\boxed{{}}" not in rewritten


def test_math_eval_reward_accepts_single_double_and_legacy_answer_formats():
    single = Sample(response=r"Final: \boxed{142}", label={"ground_truth": "142.0"})
    double = Sample(response=r"Final: \boxed{{142}}", label={"ground_truth": "142.0"})
    legacy_answer = Sample(response="Answer: 142", label={"ground_truth": "142.0"})

    assert asyncio.run(math_eval_reward_func(None, single)) == 1.0
    assert asyncio.run(math_eval_reward_func(None, double)) == 1.0
    assert asyncio.run(math_eval_reward_func(None, legacy_answer)) == 1.0


def test_write_eval_rollouts_writes_one_json_record_per_generation(tmp_path: Path):
    samples = [
        Sample(
            index=10 + i,
            prompt=f"prompt-{i // 2}",
            response=rf"reasoning \boxed{{{i}}}<|im_end|>",
            response_length=100 + i,
            label={"ground_truth": str(i)},
            status=Sample.Status.TRUNCATED if i == 1 else Sample.Status.COMPLETED,
        )
        for i in range(4)
    ]
    output_path = tmp_path / "rollouts.txt"

    _write_eval_rollouts(
        {
            "AMC23": {
                "samples": samples,
                "rewards": [1.0, 0.0, 1.0, 1.0],
                "truncated": [False, True, False, False],
            }
        },
        output_path,
        group_size=2,
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert records[0] == {
        "dataset": "AMC23",
        "prompt_index": 0,
        "rollout_index": 0,
        "sample_index": 10,
        "prompt": "prompt-0",
        "response": r"reasoning \boxed{0}",
        "label": {"ground_truth": "0"},
        "reward": 1.0,
        "correct": True,
        "response_token_length": 100,
        "status": "completed",
        "truncated": False,
        "boxed_answer": "0",
    }
    assert records[1]["prompt_index"] == 0
    assert records[1]["rollout_index"] == 1
    assert records[1]["truncated"] is True
    assert records[2]["prompt_index"] == 1
    assert records[2]["rollout_index"] == 0


def test_eval_rollout_stream_resets_then_appends_each_completed_sample(monkeypatch, tmp_path: Path):
    output_path = tmp_path / "streamed-rollouts.txt"
    output_path.write_text("stale data\n", encoding="utf-8")
    monkeypatch.setenv("OPD_EVAL_ROLLOUT_PATH", str(output_path))

    _reset_eval_rollout_stream()
    assert output_path.read_text(encoding="utf-8") == ""

    first = Sample(
        index=0,
        prompt="prompt-0",
        response=r"reasoning \boxed{7}<|im_end|>",
        response_length=11,
        label={"ground_truth": "7"},
        reward=1.0,
        status=Sample.Status.COMPLETED,
    )
    second = Sample(
        index=1,
        prompt="prompt-0",
        response=r"reasoning \boxed{8}",
        response_length=12,
        label={"ground_truth": "7"},
        reward=0.0,
        status=Sample.Status.TRUNCATED,
    )

    _append_eval_rollout_stream("AMC23", first, group_size=64)
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1
    _append_eval_rollout_stream("AMC23", second, group_size=64)

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [record["sample_index"] for record in records] == [0, 1]
    assert records[0]["correct"] is True
    assert records[0]["boxed_answer"] == "7"
    assert records[1]["correct"] is False
    assert records[1]["truncated"] is True
    _close_eval_rollout_stream()
