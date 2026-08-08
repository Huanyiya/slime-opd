from argparse import Namespace
import json

from slime.rollout.on_policy_distillation import log_eval_metrics_64
from slime.utils.metric_utils import add_train_step_metric
from slime.utils.wandb_utils import _init_wandb_common


def test_add_train_step_metric_uses_existing_canonical_axis():
    metrics = {"eval/AIME24/pass@1": 0.5}

    add_train_step_metric(metrics, 40)

    assert metrics == {"eval/AIME24/pass@1": 0.5, "step": 40}


def test_wandb_eval_uses_existing_train_step_axis(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "slime.utils.wandb_utils.wandb.define_metric",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    _init_wandb_common()

    assert (("eval/*",), {"step_metric": "step"}) in calls
    assert not any(args == ("eval/step",) for args, _ in calls)


def test_log_eval_metrics_64_writes_requested_metrics(monkeypatch, tmp_path):
    result_path = tmp_path / "results.json"
    monkeypatch.setenv("OPD_EVAL_RESULT_PATH", str(result_path))
    monkeypatch.setenv("OPD_EVAL_MODEL_PATH", "/models/iter_0000179")
    args = Namespace(
        n_samples_per_eval_prompt=64,
        metric_train_step=179,
        use_wandb=False,
        use_tensorboard=False,
    )
    data = {
        "AIME24": {"rewards": [1.0] + [0.0] * 63},
        "AIME25": {"rewards": [0.0] * 64},
    }

    assert log_eval_metrics_64(0, args, data) is True

    results = json.loads(result_path.read_text())
    assert results["model_path"] == "/models/iter_0000179"
    assert results["AIME24"] == {
        "pass@1": 1 / 64,
        "pass@2": 2 / 64,
        "pass@4": 4 / 64,
        "pass@8": 8 / 64,
        "pass@16": 16 / 64,
        "pass@32": 32 / 64,
        "pass@64": 1.0,
        "mean@64": 1 / 64,
        "num_prompts": 1,
        "num_generations": 64,
    }
    assert results["AIME25"] == {
        "pass@1": 0.0,
        "pass@2": 0.0,
        "pass@4": 0.0,
        "pass@8": 0.0,
        "pass@16": 0.0,
        "pass@32": 0.0,
        "pass@64": 0.0,
        "mean@64": 0.0,
        "num_prompts": 1,
        "num_generations": 64,
    }
