from argparse import Namespace

import numpy as np
import pybase64
import pytest
import torch

from slime.rollout.on_policy_distillation import _build_teacher_payload, post_process_rewards
from slime.ray.rollout import _compute_student_topk_probability_metrics, _compute_topk_overlap_metrics
from slime.utils.types import Sample


NUM_GPUS = 0


def _args(top_k: int = 2, loss_type: str = "topk") -> Namespace:
    return Namespace(
        use_opd=True,
        opd_loss_type=loss_type,
        opd_top_k=top_k,
        reward_key=None,
    )


def _encode_array(values, dtype) -> str:
    return pybase64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("utf-8")


@pytest.mark.unit
def test_topk_teacher_payload_uses_per_position_ids_and_dummy_row():
    sample = Sample(
        tokens=[100, 101, 7, 8],
        response_length=2,
        opd_topk_token_ids=torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
    )
    payload = _build_teacher_payload(_args(), sample)
    assert payload["return_logprobs_in_base64"] is True
    assert payload["top_logprobs_num"] == 2
    assert payload["logprob_start_len"] == 1
    assert payload["token_ids_logprob"] == [11, 12, 21, 22, 0, 0]


@pytest.mark.unit
def test_sampled_teacher_payload_does_not_enable_base64():
    sample = Sample(tokens=[100, 101, 7, 8], response_length=2)
    payload = _build_teacher_payload(_args(loss_type="sampled"), sample)
    assert "return_logprobs_in_base64" not in payload
    assert "token_ids_logprob" not in payload


@pytest.mark.unit
def test_topk_teacher_base64_response_skips_empty_leading_row_and_preserves_r_by_k():
    teacher_log_probs = [[-1.1, -1.2], [-2.1, -2.2]]
    teacher_ids = [[11, 12], [21, 22]]
    teacher_topk_ids = [[12, 99], [21, 22]]
    sample = Sample(
        tokens=[100, 101, 7, 8],
        response_length=2,
        opd_topk_token_ids=torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
        reward={
            "meta_info": {
                "input_token_ids_logprobs_val_b64": _encode_array(teacher_log_probs, np.float32),
                "input_token_ids_logprobs_idx_b64": _encode_array(teacher_ids, np.int32),
                "input_token_ids_logprobs_len": [0, 2, 2],
                "input_top_logprobs_idx_b64": _encode_array(teacher_topk_ids, np.int32),
                "input_top_logprobs_k": [0, 2, 2],
            }
        },
    )
    raw_rewards, rewards = post_process_rewards(_args(), [sample])
    assert raw_rewards == [0.0]
    assert rewards == [0.0]
    torch.testing.assert_close(
        sample.opd_topk_teacher_log_probs,
        torch.tensor([[-1.1, -1.2], [-2.1, -2.2]], dtype=torch.float32),
    )
    assert sample.metadata["topk_overlap_count"] == 3
    assert sample.metadata["topk_overlap_total"] == 4
    assert sample.metadata["top1_overlap_count"] == 1
    assert sample.metadata["top1_overlap_total"] == 2
    assert sample.metadata["teacher_top1_in_student_topk_count"] == 2
    assert sample.metadata["teacher_top1_in_student_topk_total"] == 2
    assert sample.metadata["teacher_top1_in_student_rank_sum"] == 3
    assert sample.metadata["teacher_top1_in_student_rank_total"] == 2
    assert _compute_topk_overlap_metrics(_args(), [sample]) == {
        "teacher_top1_in_student_rank_mean": 1.5,
        "teacher_top1_in_student_topk_ratio": 1.0,
        "top1_overlap_ratio": 0.5,
        "topk_overlap_ratio": 0.75,
    }


@pytest.mark.unit
def test_topk_teacher_base64_response_rejects_wrong_decoded_size():
    sample = Sample(
        tokens=[100, 101, 7, 8],
        response_length=2,
        opd_topk_token_ids=torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
        reward={
            "meta_info": {
                "input_token_ids_logprobs_val_b64": _encode_array([[-1.1, -1.2]], np.float32),
                "input_token_ids_logprobs_idx_b64": _encode_array([[11, 12], [21, 22]], np.int32),
                "input_token_ids_logprobs_len": [0, 2, 2],
                "input_top_logprobs_idx_b64": _encode_array([[11, 12], [21, 22]], np.int32),
                "input_top_logprobs_k": [0, 2, 2],
            }
        },
    )
    with pytest.raises(ValueError, match="decoded bytes"):
        post_process_rewards(_args(), [sample])


@pytest.mark.unit
def test_topk_overlap_uses_loss_mask_and_aggregates_by_token():
    def make_sample(loss_mask, teacher_topk_ids):
        sample = Sample(
            tokens=[100, 101, 7, 8],
            response_length=2,
            loss_mask=loss_mask,
            opd_topk_token_ids=torch.tensor([[11, 12], [21, 22]], dtype=torch.int32),
            reward={
                "meta_info": {
                    "input_token_ids_logprobs_val_b64": _encode_array(
                        [[-1.1, -1.2], [-2.1, -2.2]], np.float32
                    ),
                    "input_token_ids_logprobs_idx_b64": _encode_array(
                        [[11, 12], [21, 22]], np.int32
                    ),
                    "input_token_ids_logprobs_len": [0, 2, 2],
                    "input_top_logprobs_idx_b64": _encode_array(teacher_topk_ids[:2], np.int32),
                    "input_top_logprobs_k": [0, 2, 2],
                }
            },
        )
        post_process_rewards(_args(), [sample])
        return sample

    masked_sample = make_sample([1, 0], [[12, 99], [21, 22], [31, 32]])
    full_sample = make_sample([1, 1], [[11, 12], [98, 99], [31, 32]])

    assert masked_sample.metadata["topk_overlap_count"] == 1
    assert masked_sample.metadata["topk_overlap_total"] == 2
    assert full_sample.metadata["topk_overlap_count"] == 2
    assert full_sample.metadata["topk_overlap_total"] == 4
    assert _compute_topk_overlap_metrics(_args(), [masked_sample, full_sample]) == {
        "teacher_top1_in_student_rank_mean": 2.0,
        "teacher_top1_in_student_topk_ratio": pytest.approx(2 / 3),
        "top1_overlap_ratio": pytest.approx(1 / 3),
        "topk_overlap_ratio": 0.5
    }


@pytest.mark.unit
def test_topk_overlap_metrics_include_top1_top2_top4_for_k16():
    sample = Sample(
        metadata={
            "top1_overlap_count": 1,
            "top1_overlap_total": 2,
            "top2_overlap_count": 3,
            "top2_overlap_total": 4,
            "top4_overlap_count": 5,
            "top4_overlap_total": 8,
            "topk_overlap_count": 20,
            "topk_overlap_total": 32,
            "teacher_top1_in_student_topk_count": 1,
            "teacher_top1_in_student_topk_total": 2,
            "teacher_top1_in_student_rank_sum": 18,
            "teacher_top1_in_student_rank_total": 2,
        }
    )
    assert _compute_topk_overlap_metrics(_args(top_k=16), [sample]) == {
        "teacher_top1_in_student_rank_mean": 9.0,
        "teacher_top1_in_student_topk_ratio": 0.5,
        "top1_overlap_ratio": 0.5,
        "top2_overlap_ratio": 0.75,
        "top4_overlap_ratio": 0.625,
        "topk_overlap_ratio": 0.625,
    }


@pytest.mark.unit
def test_teacher_top1_missing_from_student_top16_uses_rank_17():
    top_k = 16
    student_ids = np.arange(top_k, dtype=np.int32).reshape(1, top_k)
    teacher_topk_ids = np.concatenate(
        [
            np.asarray([[99, *range(20, 35)]], dtype=np.int32),
            np.arange(100, 100 + top_k, dtype=np.int32).reshape(1, top_k),
        ],
        axis=0,
    )
    sample = Sample(
        tokens=[100, 7],
        response_length=1,
        opd_topk_token_ids=torch.from_numpy(student_ids.copy()),
        reward={
            "meta_info": {
                "input_token_ids_logprobs_val_b64": _encode_array(np.zeros((1, top_k)), np.float32),
                "input_token_ids_logprobs_idx_b64": _encode_array(student_ids, np.int32),
                "input_token_ids_logprobs_len": [0, top_k],
                "input_top_logprobs_idx_b64": _encode_array(teacher_topk_ids[:1], np.int32),
                "input_top_logprobs_k": [0, top_k],
            }
        },
    )

    post_process_rewards(_args(top_k=top_k), [sample])

    assert sample.metadata["teacher_top1_in_student_topk_count"] == 0
    assert sample.metadata["teacher_top1_in_student_topk_total"] == 1
    assert sample.metadata["teacher_top1_in_student_rank_sum"] == 17
    assert sample.metadata["teacher_top1_in_student_rank_total"] == 1
    metrics = _compute_topk_overlap_metrics(_args(top_k=top_k), [sample])
    assert metrics["teacher_top1_in_student_topk_ratio"] == 0.0
    assert metrics["teacher_top1_in_student_rank_mean"] == 17.0


@pytest.mark.unit
def test_student_topk_probability_metrics_are_weighted_by_valid_rollout_tokens():
    samples = [
        Sample(
            metadata={
                "student_topk_prob_sum_numerator": 1.6,
                "student_topk_prob_sum_denominator": 2,
                "student_top4_prob_sum_numerator": 1.2,
                "student_top4_prob_sum_denominator": 2,
                "student_top1_prob_numerator": 0.8,
                "student_top1_prob_denominator": 2,
            }
        ),
        Sample(
            metadata={
                "student_topk_prob_sum_numerator": 0.6,
                "student_topk_prob_sum_denominator": 1,
                "student_top4_prob_sum_numerator": 0.4,
                "student_top4_prob_sum_denominator": 1,
                "student_top1_prob_numerator": 0.2,
                "student_top1_prob_denominator": 1,
            }
        ),
    ]

    assert _compute_student_topk_probability_metrics(_args(top_k=16), samples) == {
        "student_topk_prob_sum": pytest.approx(2.2 / 3),
        "student_top4_prob_sum": pytest.approx(1.6 / 3),
        "student_top1_prob": pytest.approx(1.0 / 3),
    }


@pytest.mark.unit
def test_sampled_teacher_json_response_is_unchanged():
    sample = Sample(
        tokens=[100, 101, 7, 8],
        response_length=2,
        reward={
            "meta_info": {
                "input_token_logprobs": [[None, 100, None], [-0.1, 101, None], [-0.2, 7, None], [-0.3, 8, None]]
            }
        },
    )
    post_process_rewards(_args(loss_type="sampled"), [sample])
    torch.testing.assert_close(sample.teacher_log_probs, torch.tensor([-0.2, -0.3], dtype=torch.float32))
