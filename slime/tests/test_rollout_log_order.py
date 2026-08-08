from argparse import Namespace

import slime.ray.rollout as rollout_module
from slime.utils.types import Sample
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS


class _FakeServer:
    def __init__(self):
        self.calls = []

    def offload(self):
        self.calls.append("offload")

    def onload(self, tags=None):
        self.calls.append(("onload", tags))

    def onload_weights(self):
        self.calls.append("onload_weights")

    def onload_kv(self):
        self.calls.append("onload_kv")


def test_training_rollout_logs_after_reward_post_process(monkeypatch):
    manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager.args = Namespace(
        ci_test=False,
        use_fault_tolerance=False,
        log_opd_phase_times_only=False,
        debug_rollout_only=False,
    )
    sample = Sample(metadata={})
    events = []

    manager.health_monitoring_resume = lambda: None
    manager._get_rollout_data = lambda rollout_id: ([sample], None)
    manager._save_debug_rollout_data = lambda data, rollout_id, evaluation: None

    def convert(samples):
        events.append("convert")
        samples[0].metadata["topk_overlap_total"] = 1
        return {"converted": True}

    manager._convert_samples_to_train_data = convert
    manager._split_train_data_by_dp = lambda train_data: (events.append("split"), train_data)[1]

    def log_rollout_data(rollout_id, args, samples, metrics, rollout_time, metric_step=None):
        events.append("log")
        assert samples[0].metadata["topk_overlap_total"] == 1

    monkeypatch.setattr(rollout_module, "_log_rollout_data", log_rollout_data)

    assert manager.generate(rollout_id=3) == {"converted": True}
    assert events == ["convert", "log", "split"]


def test_sequential_opd_residency_ignores_duplicate_phase_transitions():
    manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager._student_server = _FakeServer()
    manager._teacher_server = _FakeServer()
    manager._sequential_residency = None
    manager.health_monitoring_pause = lambda: None

    # Startup and the outer train loop may both ask for offload. Once no model
    # is resident, the second request must not call the stateful SGLang memory
    # saver again.
    manager.offload()
    manager.offload()
    assert manager._student_server.calls == []
    assert manager._teacher_server.calls == []

    manager.onload_weights()
    manager.onload_weights()
    manager.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])
    manager.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])
    assert manager._student_server.calls == ["onload_weights", "onload_kv"]
    assert manager._sequential_residency == "student"

    manager.offload()
    manager.offload()
    assert manager._student_server.calls == ["onload_weights", "onload_kv", "offload"]
    assert manager._sequential_residency is None

    # Teacher scoring leaves the teacher resident only until its own finally
    # block. A later outer offload must likewise be harmless.
    manager._sequential_residency = "teacher"
    manager.offload()
    manager.offload()
    assert manager._teacher_server.calls == ["offload"]
    assert manager._sequential_residency is None

    manager.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])
    manager.onload()
    assert manager._student_server.calls[-2:] == ["onload_weights", "onload_kv"]
    assert manager._sequential_residency == "student"


def test_sequential_opd_kv_onload_requires_student_weights():
    manager_cls = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_cls)
    manager._student_server = _FakeServer()
    manager._teacher_server = _FakeServer()
    manager._sequential_residency = None

    try:
        manager.onload_kv()
    except RuntimeError as exc:
        assert "requires student weights" in str(exc)
    else:
        raise AssertionError("onload_kv() should reject a missing student-weight phase")

