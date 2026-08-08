import sys
import types
from argparse import Namespace

import pytest
import torch


NUM_GPUS = 0


@pytest.mark.unit
@pytest.mark.parametrize("opd_kl_loss_type", ["k1", "k3"])
def test_opd_kl_estimators_match_formula_in_topk_and_sampled_paths(monkeypatch, opd_kl_loss_type):
    previous_loss = sys.modules.pop("slime.backends.megatron_utils.loss", None)
    previous_cp_utils = sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_data_parallel_world_size=lambda with_context_parallel=True: 1,
    )
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = mpu_stub
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)

    try:
        from slime.backends.megatron_utils.loss import apply_opd_kl_to_advantages, loss_function

        args = Namespace(
            use_opd=True,
            opd_loss_type="topk",
            opd_top_k=2,
            opd_kl_coef=1.7,
            opd_kl_loss_type=opd_kl_loss_type,
            loss_type="policy_loss",
            rollout_temperature=1.0,
            log_probs_chunk_size=-1,
            allgather_cp=False,
            calculate_per_token_loss=False,
            recompute_loss_function=False,
        )
        logits = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [1.0, 0.0, -1.0, 2.0],
                    [-0.5, 1.5, 0.5, 0.0],
                    [0.2, 0.1, 0.0, -0.1],
                ]
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        topk_ids = torch.tensor([[3, 0], [1, 2]], dtype=torch.long)
        teacher_log_probs = torch.tensor([[-0.8, -2.2], [-0.4, -1.7]], dtype=torch.float32)
        batch = {
            "unconcat_tokens": [torch.tensor([10, 11, 12, 13])],
            "total_lengths": [4],
            "response_lengths": [2],
            "loss_masks": [torch.tensor([1, 1], dtype=torch.int32)],
            "rollout_mask_sums": torch.tensor([2.0]),
            "opd_topk_token_ids": [topk_ids],
            "opd_topk_teacher_log_probs": [teacher_log_probs],
        }

        loss, _normalizer, metrics = loss_function(
            args,
            batch,
            num_microbatches=1,
            step_global_batch_size=1,
            logits=logits,
        )

        reference_logits = logits.detach().clone().requires_grad_()
        student_log_probs = torch.log_softmax(reference_logits[0, 1:3], dim=-1).gather(-1, topk_ids)
        weights = torch.softmax(student_log_probs, dim=-1)
        d = student_log_probs - teacher_log_probs
        per_candidate_kl = d if opd_kl_loss_type == "k1" else d + torch.exp(-d) - 1
        reference_topk_loss = (weights * per_candidate_kl).sum(-1).mean()
        reference_loss = args.opd_kl_coef * reference_topk_loss
        reference_log_probs = torch.log_softmax(reference_logits[0, 1:3], dim=-1)
        reference_entropy = -(reference_log_probs.exp() * reference_log_probs).sum(-1).mean()

        torch.testing.assert_close(loss, reference_loss, rtol=1e-6, atol=1e-7)
        assert metrics["keys"] == ["loss", "opd_topk_loss", "student_entropy"]
        torch.testing.assert_close(metrics["values"][3], reference_entropy, rtol=1e-6, atol=1e-7)
        expected_token_kls = (weights * per_candidate_kl).sum(dim=-1)
        expected_token_variance = expected_token_kls.var(unbiased=False)
        torch.testing.assert_close(
            metrics["_opd_token_kl_variances"],
            expected_token_variance.reshape(1),
            rtol=1e-6,
            atol=1e-7,
        )

        loss.backward()
        reference_loss.backward()
        torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-5, atol=1e-7)

        sampled_student_log_probs = [torch.tensor([-0.4, -1.8], dtype=torch.float32)]
        sampled_teacher_log_probs = [torch.tensor([-0.7, -1.2], dtype=torch.float32)]
        sampled_advantages = [torch.tensor([0.5, -0.3], dtype=torch.float32)]
        sampled_rollout_data = {"teacher_log_probs": sampled_teacher_log_probs}
        apply_opd_kl_to_advantages(
            args,
            sampled_rollout_data,
            sampled_advantages,
            sampled_student_log_probs,
        )
        sampled_d = sampled_student_log_probs[0] - sampled_teacher_log_probs[0]
        sampled_kl = sampled_d if opd_kl_loss_type == "k1" else sampled_d + torch.exp(-sampled_d) - 1
        torch.testing.assert_close(
            sampled_advantages[0],
            torch.tensor([0.5, -0.3]) - args.opd_kl_coef * sampled_kl,
        )
        torch.testing.assert_close(sampled_rollout_data["opd_reverse_kl"][0], sampled_kl)
    finally:
        if previous_loss is None:
            sys.modules.pop("slime.backends.megatron_utils.loss", None)
        else:
            sys.modules["slime.backends.megatron_utils.loss"] = previous_loss
        if previous_cp_utils is None:
            sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
        else:
            sys.modules["slime.backends.megatron_utils.cp_utils"] = previous_cp_utils


@pytest.mark.unit
def test_k1_delta_clip_and_metrics_use_topk_entry_denominator(monkeypatch):
    previous_loss = sys.modules.pop("slime.backends.megatron_utils.loss", None)
    previous_cp_utils = sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_data_parallel_world_size=lambda with_context_parallel=True: 1,
    )
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = mpu_stub
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)

    try:
        from slime.backends.megatron_utils.loss import _append_k1_opd_clip_metrics, _clip_k1_opd_delta

        args = Namespace(opd_kl_loss_type="k1", opd_k1_diff_clip=1.0)
        raw_delta = torch.tensor([[2.0, -0.5], [-3.0, 0.25]])
        torch.testing.assert_close(_clip_k1_opd_delta(raw_delta, args), torch.tensor([[1.0, -0.5], [-1.0, 0.25]]))

        metrics = {}
        _append_k1_opd_clip_metrics(metrics, raw_delta, torch.tensor([True, False]), args)
        assert metrics["opd_k1_diff_clip_ratio"] == 1.0
        assert metrics["opd_k1_diff_preclip_min"] == -0.5
        assert metrics["opd_k1_diff_preclip_max"] == 2.0
        # One valid response position has K=2 entries, so the denominator is 2.
        assert metrics["_metric_reduce_denominators"][0] == 2.0
        assert metrics["_metric_reduce_ops"].tolist() == [3, 1, 2]

        k3_args = Namespace(opd_kl_loss_type="k3", opd_k1_diff_clip=1.0)
        torch.testing.assert_close(_clip_k1_opd_delta(raw_delta, k3_args), raw_delta)
    finally:
        if previous_loss is None:
            sys.modules.pop("slime.backends.megatron_utils.loss", None)
        else:
            sys.modules["slime.backends.megatron_utils.loss"] = previous_loss
        if previous_cp_utils is None:
            sys.modules.pop("slime.backends.megatron_utils.cp_utils", None)
        else:
            sys.modules["slime.backends.megatron_utils.cp_utils"] = previous_cp_utils
