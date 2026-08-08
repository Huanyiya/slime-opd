"""CPU-only lifecycle tests for colocated sequential OPD.

These tests deliberately exercise orchestration boundaries rather than model
math.  The production job shares the same eight GPUs between student SGLang,
teacher SGLang, and Megatron, so an ordering regression can fail before the
first rollout even when each component works in isolation.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# The production Ray runtime sets PYTHONPATH to this checkout explicitly.
# Mirror that contract so this CPU-only test imports the same Megatron code.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Megatron-LM"))

import train as train_module
from slime.backends.megatron_utils import actor as actor_module
from slime.backends.megatron_utils import data as data_module
from slime.backends.sglang_utils import sglang_engine as sglang_engine_module
from slime.ray import actor_group as actor_group_module
from slime.ray import rollout as rollout_module


_ROLLOUT_MANAGER_IMPL = rollout_module.RolloutManager.__ray_metadata__.modified_class


def _call_rollout_manager(method_name, manager, *args, **kwargs):
    """Call the implementation hidden below Ray's actor/tracing wrappers."""
    return getattr(_ROLLOUT_MANAGER_IMPL, method_name).__wrapped__(manager, *args, **kwargs)


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeRolloutManager:
    def __init__(self, events):
        self.onload_weights = _RemoteMethod(lambda: events.append("student_weights_onload"))
        self.onload_kv = _RemoteMethod(lambda: events.append("student_kv_onload"))
        self.offload = _RemoteMethod(lambda: events.append("student_offload"))
        self.dispose = _RemoteMethod(lambda: events.append("rollout_dispose"))
        self.generate = _RemoteMethod(self._generate)
        self.events = events

    def _generate(self, rollout_id, metric_step=None):
        self.events.extend(["student_rollout", "student_offload_for_teacher", "teacher_forward", "teacher_offload"])
        return "rollout-data"


class _FakeActorGroup:
    def __init__(self, events):
        self.events = events
        self.update_count = 0

    def update_weights(self):
        self.update_count += 1
        self.events.append(f"actor_disk_sync_{self.update_count}")
        return {}

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.events.append("actor_train")
        return [None]

    def clear_memory(self):
        raise AssertionError("offloaded actor must not use the non-offload clear_memory path")


def test_one_sequential_opd_iteration_has_exclusive_gpu_phase_order(monkeypatch):
    """Lock down startup sync and one complete 8-GPU sequential iteration."""

    events = []
    rollout_manager = _FakeRolloutManager(events)
    actor_group = _FakeActorGroup(events)
    args = SimpleNamespace(
        release_train=False,
        offload_rollout=True,
        offload_train=True,
        log_opd_phase_times_only=False,
        check_weight_update_equal=False,
        start_rollout_id=0,
        num_rollout=1,
        eval_interval=None,
        skip_eval_before_train=True,
        use_critic=False,
        num_critic_only_steps=0,
        save_interval=None,
        rollout_global_dataset=False,
    )

    monkeypatch.setattr(train_module.ray, "get", lambda value: value)
    monkeypatch.setattr(train_module, "configure_logger", lambda: None)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(train_module, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_module, "finish_tracking", lambda _args: events.append("tracking_finish"))
    monkeypatch.setattr(
        train_module,
        "create_rollout_manager",
        lambda _args, _pg: (rollout_manager, None),
    )
    monkeypatch.setattr(
        train_module,
        "create_training_models",
        lambda _args, _pgs, _manager: (actor_group, None),
    )
    monkeypatch.setattr(train_module, "compute_train_steps_per_rollout", lambda _args: 1)
    monkeypatch.setattr(train_module, "compute_train_step_before_rollout", lambda _args, _rid: 0)
    monkeypatch.setattr(train_module, "compute_first_train_step_for_rollout", lambda _args, _rid: 1)
    monkeypatch.setattr(train_module, "should_run_periodic_action", lambda *args, **kwargs: False)

    train_module.train(args)

    assert events == [
        # Startup: load only student weights, sync actor checkpoint, then KV.
        "student_weights_onload",
        "actor_disk_sync_1",
        "student_kv_onload",
        # Rollout and frozen-teacher forward never overlap residency.
        "student_rollout",
        "student_offload_for_teacher",
        "teacher_forward",
        "teacher_offload",
        # The outer offload is idempotent, then Megatron exclusively trains.
        "student_offload",
        "actor_train",
        # Publish updated actor and prepare the next student rollout.
        "student_weights_onload",
        "actor_disk_sync_2",
        "student_kv_onload",
        "rollout_dispose",
        "tracking_finish",
    ]


def test_disk_sync_uses_cached_topology_without_nccl_or_actor_weights(monkeypatch):
    """Disk export must keep both NCCL and CUDA actor memory paused."""

    events = []

    class _Updater:
        def update_weights(self):
            events.append("write_cpu_checkpoint")

    fake_actor = SimpleNamespace(
        args=SimpleNamespace(
            debug_train_only=False,
            debug_rollout_only=False,
            use_fault_tolerance=False,
            offload_train=True,
            use_critic=False,
            colocate=True,
            update_weight_transport="disk",
            keep_old_actor=False,
        ),
        rollout_manager=SimpleNamespace(
            get_updatable_engines_and_lock=_RemoteMethod(
                lambda: ([object()], object(), 0, [1], [0], [{}])
            )
        ),
        weight_updater=_Updater(),
    )

    monkeypatch.setattr(actor_module.ray, "get", lambda value: value)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        actor_module,
        "reload_process_groups",
        lambda: (_ for _ in ()).throw(AssertionError("disk sync must not restore NCCL")),
    )
    monkeypatch.setattr(actor_module, "destroy_process_groups", lambda: events.append("destroy_topology"))
    monkeypatch.setattr(actor_module, "print_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        actor_module.torch_memory_saver,
        "disable",
        lambda: (_ for _ in ()).throw(AssertionError("disk sync must not resume CUDA actor weights")),
    )

    # Bypass only the timing decorator; run the real update_weights body.
    actor_module.MegatronTrainRayActor.update_weights.__wrapped__(fake_actor)

    assert events == ["write_cpu_checkpoint", "destroy_topology"]


def test_actor_wake_waits_for_h2d_before_reloading_nccl(monkeypatch):
    events = []
    fake_actor = SimpleNamespace(
        args=SimpleNamespace(offload_train=True),
        role="actor",
        _active_model_tag="ref",
        _train_residency="cpu",
        _switch_model=lambda name: events.append(f"switch_{name}"),
    )

    monkeypatch.setattr(actor_module, "print_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "clear_memory", lambda: events.append("clear"))
    monkeypatch.setattr(actor_module, "reload_process_groups", lambda: events.append("reload_nccl"))
    monkeypatch.setattr(actor_module.torch_memory_saver, "resume", lambda: events.append("resume_h2d"))
    monkeypatch.setattr(actor_module.torch.cuda, "synchronize", lambda: events.append("cuda_sync"))

    actor_module.MegatronTrainRayActor.wake_up.__wrapped__(fake_actor)

    assert events == ["resume_h2d", "cuda_sync", "clear", "reload_nccl", "switch_actor"]
    assert fake_actor._train_residency == "gpu"


def test_actor_wake_skips_restore_when_actor_is_already_active(monkeypatch):
    events = []
    fake_actor = SimpleNamespace(
        args=SimpleNamespace(offload_train=True),
        role="actor",
        _active_model_tag="actor",
        _train_residency="cpu",
        _switch_model=lambda name: events.append(f"switch_{name}"),
    )

    monkeypatch.setattr(actor_module, "print_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "clear_memory", lambda: events.append("clear"))
    monkeypatch.setattr(actor_module, "reload_process_groups", lambda: events.append("reload_nccl"))
    monkeypatch.setattr(actor_module.torch_memory_saver, "resume", lambda: events.append("resume_h2d"))
    monkeypatch.setattr(actor_module.torch.cuda, "synchronize", lambda: events.append("cuda_sync"))

    actor_module.MegatronTrainRayActor.wake_up.__wrapped__(fake_actor)

    assert events == ["resume_h2d", "cuda_sync", "clear", "reload_nccl"]
    assert fake_actor._train_residency == "gpu"


def test_actor_sleep_moves_gpu_state_to_cpu_after_copy_completion(monkeypatch):
    events = []
    fake_actor = SimpleNamespace(
        args=SimpleNamespace(offload_train=True, use_critic=False, colocate=True),
        role="actor",
        _train_residency="gpu",
    )

    monkeypatch.setattr(actor_module, "clear_memory", lambda **_kwargs: events.append("clear"))
    monkeypatch.setattr(actor_module, "print_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actor_module, "destroy_process_groups", lambda: events.append("destroy_nccl"))
    monkeypatch.setattr(actor_module.torch_memory_saver, "pause", lambda: events.append("pause_d2h"))
    monkeypatch.setattr(actor_module.torch.cuda, "synchronize", lambda: events.append("cuda_sync"))

    actor_module.MegatronTrainRayActor.sleep.__wrapped__(fake_actor)

    assert events == ["clear", "destroy_nccl", "pause_d2h", "cuda_sync"]
    assert fake_actor._train_residency == "cpu"


def test_ray_cpu_tensors_use_blocking_h2d_copy():
    calls = []

    class _Tensor:
        def to(self, **kwargs):
            calls.append(kwargs)
            return "gpu-tensor"

    result = data_module.to_device_blocking(_Tensor(), device=3, dtype="float32")

    assert result == "gpu-tensor"
    assert calls == [{"device": 3, "dtype": "float32", "non_blocking": False}]


def test_sequential_student_teacher_student_residency_cycle(monkeypatch):
    events = []

    class _Server:
        def __init__(self, name):
            self.name = name

        def onload_weights(self):
            events.append(f"{self.name}_weights_gpu")

        def onload_kv(self):
            events.append(f"{self.name}_kv_gpu")

        def onload(self):
            events.append(f"{self.name}_gpu")

        def offload(self):
            events.append(f"{self.name}_cpu")

    student = _Server("student")
    teacher = _Server("teacher")
    manager = SimpleNamespace(
        _student_server=student,
        _teacher_server=teacher,
        _sequential_residency=None,
        _teacher_generate_urls=["http://teacher-0/generate", "http://teacher-1/generate"],
        args=SimpleNamespace(sglang_server_concurrency=2),
    )
    manager._offload_sequential_active_server = lambda: _call_rollout_manager(
        "_offload_sequential_active_server", manager
    )

    async def fake_rm(_args, sample, rm_url):
        events.append(f"score:{rm_url}")
        return sample.expected_reward

    monkeypatch.setattr(rollout_module, "async_rm", fake_rm)
    monkeypatch.setattr(rollout_module, "run", asyncio.run)

    _call_rollout_manager("onload_weights", manager)
    assert manager._sequential_residency == "student_weights"
    _call_rollout_manager("onload_kv", manager)
    assert manager._sequential_residency == "student"

    samples = [SimpleNamespace(expected_reward=1.0, reward=None), SimpleNamespace(expected_reward=2.0, reward=None)]
    _call_rollout_manager("_score_with_sequential_teacher", manager, samples)

    assert manager._sequential_residency is None
    assert [sample.reward for sample in samples] == [1.0, 2.0]

    _call_rollout_manager("onload_weights", manager)
    _call_rollout_manager("onload_kv", manager)
    assert manager._sequential_residency == "student"
    assert events == [
        "student_weights_gpu",
        "student_kv_gpu",
        "student_cpu",
        "teacher_gpu",
        "score:http://teacher-0/generate",
        "score:http://teacher-1/generate",
        "teacher_cpu",
        "student_weights_gpu",
        "student_kv_gpu",
    ]


def test_sequential_frozen_teacher_keeps_cpu_weight_backup(monkeypatch):
    """A frozen teacher must survive release/resume without a disk reload."""

    captured = {}
    engine = SimpleNamespace(get_url=_RemoteMethod(lambda: "http://teacher-0"))
    server = SimpleNamespace(
        router_ip="127.0.0.1",
        router_port=15100,
        engines=[engine],
        model_name="default",
        update_weights=True,
    )

    def fake_start(teacher_args, _pg):
        captured["teacher_args"] = teacher_args
        return {"default": server}, []

    monkeypatch.setattr(rollout_module, "start_rollout_servers", fake_start)
    monkeypatch.setattr(rollout_module.ray, "get", lambda value: value)

    args = SimpleNamespace(
        sequential_opd_teacher_model_path="/teacher",
        sequential_opd_teacher_num_gpus_per_engine=1,
        sequential_opd_teacher_mem_fraction_static=0.7,
        sequential_opd_teacher_chunked_prefill_size=6144,
        sglang_model_routers={},
    )
    manager = SimpleNamespace(
        args=args,
        _student_server=object(),
        _teacher_server=None,
        _sequential_residency="student",
        servers={},
    )

    def fake_offload():
        manager._sequential_residency = None

    manager._offload_sequential_active_server = fake_offload
    _call_rollout_manager("_init_sequential_opd_teacher", manager, object())

    assert captured["teacher_args"].sglang_enable_weights_cpu_backup is True
    assert manager._sequential_residency is None


def test_sequential_rollout_rejects_partial_or_wrong_gpu_residency():
    manager = SimpleNamespace(
        _teacher_server=object(),
        _sequential_residency="student_weights",
    )
    with pytest.raises(RuntimeError, match="complete student weights/KV"):
        _call_rollout_manager("generate", manager, rollout_id=0)

    manager._sequential_residency = "teacher"
    with pytest.raises(RuntimeError, match="student weights on GPU"):
        _call_rollout_manager("get_updatable_engines_and_lock", manager)


def test_disk_reload_checks_every_student_engine_weight_version(monkeypatch, tmp_path):
    events = []

    class _Engine:
        def __init__(self, index, version):
            self.index = index
            self.version = version
            self.pause_generation = _RemoteMethod(lambda: events.append(f"pause:{index}"))
            self.flush_cache = _RemoteMethod(lambda: events.append(f"flush:{index}"))
            self.update_weights_from_disk = _RemoteMethod(self._update)
            self.get_weight_version = _RemoteMethod(lambda: self.version)
            self.continue_generation = _RemoteMethod(lambda: events.append(f"continue:{index}"))

        def _update(self, model_path, weight_version):
            events.append(f"load:{self.index}:{Path(model_path).name}:{weight_version}")

    engines = [_Engine(index, "7") for index in range(8)]
    manager = SimpleNamespace(
        get_updatable_engines_and_lock=_RemoteMethod(lambda: (engines, object(), 0, [], [], [])),
    )
    group = SimpleNamespace(
        args=SimpleNamespace(
            offload_rollout=False,
            update_weight_local_checkpoint_dir=None,
            update_weight_disk_keep_files=True,
        ),
        _rollout_manager=manager,
    )
    monkeypatch.setattr(actor_group_module.ray, "get", lambda value: value)

    actor_group_module.RayTrainGroup._reload_rollout_weights_from_disk(group, tmp_path / "weight_v000007", "7")

    assert [event for event in events if event.startswith("load:")] == [
        f"load:{index}:weight_v000007:7" for index in range(8)
    ]
    assert [event for event in events if event.startswith("continue:")] == [
        f"continue:{index}" for index in range(8)
    ]

    engines[-1].version = "6"
    with pytest.raises(RuntimeError, match="engine 7: 6"):
        actor_group_module.RayTrainGroup._reload_rollout_weights_from_disk(
            group, tmp_path / "weight_v000007", "7"
        )


def test_sglang_weight_version_uses_supported_model_info_endpoint(monkeypatch):
    calls = []

    class _Response:
        def raise_for_status(self):
            calls.append("status_ok")

        def json(self):
            return {"weight_version": "9"}

    monkeypatch.setattr(
        sglang_engine_module.requests,
        "get",
        lambda url: calls.append(url) or _Response(),
    )
    engine = sglang_engine_module.SGLangEngine.__new__(sglang_engine_module.SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 15000

    assert engine.get_weight_version() == "9"
    assert calls == ["http://127.0.0.1:15000/model_info", "status_ok"]
