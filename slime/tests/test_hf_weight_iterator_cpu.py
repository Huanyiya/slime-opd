from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.update_weight import hf_weight_iterator_direct as iterator_module
from slime.backends.megatron_utils.update_weight import update_weight_from_disk as disk_module


def test_get_megatron_full_params_keeps_cpu_backup_off_cuda(monkeypatch):
    monkeypatch.setattr(iterator_module, "monkey_patch_torch_reductions", lambda: None)
    monkeypatch.setattr(iterator_module.dist, "get_rank", lambda: 0)
    for name in (
        "get_pipeline_model_parallel_world_size",
        "get_expert_model_parallel_world_size",
        "get_tensor_model_parallel_world_size",
        "get_expert_tensor_parallel_world_size",
    ):
        monkeypatch.setattr(
            iterator_module.mpu,
            name,
            lambda: (_ for _ in ()).throw(AssertionError("disk sync must use cached topology")),
        )
    monkeypatch.setattr(
        iterator_module.torch.cuda,
        "current_device",
        lambda: (_ for _ in ()).throw(AssertionError("CPU disk sync must not query CUDA")),
    )
    monkeypatch.setattr(
        iterator_module.torch.cuda,
        "synchronize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("CPU disk sync must not synchronize CUDA")),
    )

    info = SimpleNamespace(
        name="weight",
        src_rank=0,
        shape=torch.Size([2, 2]),
        dtype=torch.float32,
        attrs={"tensor_model_parallel": False},
    )
    cpu_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    result = iterator_module._get_megatron_full_params(
        [info],
        {"weight": cpu_weight},
        target_device=torch.device("cpu"),
        parallel_sizes={"pipeline": 1, "expert_model": 1, "tensor": 1, "expert_tensor": 1},
    )

    assert len(result) == 1
    assert result[0].device.type == "cpu"
    torch.testing.assert_close(result[0], cpu_weight)


def test_offloaded_disk_sync_rejects_nccl_before_accessing_weights(monkeypatch):
    updater = object.__new__(disk_module.UpdateWeightFromDisk)
    updater.args = SimpleNamespace(offload_train=True)
    updater.weight_version = 0
    monkeypatch.setattr(disk_module.dist, "get_backend", lambda: "nccl")

    with pytest.raises(RuntimeError, match="temporary Gloo WORLD"):
        updater.update_weights()

    assert updater.weight_version == 0
