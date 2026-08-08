import types

import pytest

from slime.utils.eval_config import build_eval_dataset_configs


@pytest.mark.unit
def test_eval_cli_sampling_overrides_are_applied():
    args = types.SimpleNamespace(
        n_samples_per_eval_prompt=16,
        n_samples_per_prompt=4,
        eval_temperature=0.7,
        rollout_temperature=1.0,
        eval_top_p=0.8,
        rollout_top_p=1.0,
        eval_top_k=20,
        rollout_top_k=-1,
        eval_min_p=0.0,
        eval_presence_penalty=1.5,
        eval_repetition_penalty=1.0,
        eval_max_response_len=8192,
        rollout_max_response_len=4096,
        eval_min_new_tokens=7,
        eval_input_key="prompt",
        input_key="input",
        eval_label_key="reward_model",
        label_key="label",
        eval_tool_key=None,
        tool_key=None,
        metadata_key=None,
        multimodal_keys=None,
        apply_chat_template=True,
        apply_chat_template_kwargs={"enable_thinking": False},
        eval_custom_rm_path="custom.eval_rm",
        custom_rm_path=None,
    )

    configs = build_eval_dataset_configs(args, [{"name": "AIME24", "path": "test.parquet"}], {})

    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.8
    assert cfg.top_k == 20
    assert cfg.min_p == 0.0
    assert cfg.presence_penalty == 1.5
    assert cfg.repetition_penalty == 1.0
    assert cfg.max_response_len == 8192
    assert cfg.min_new_tokens == 7
    assert cfg.apply_chat_template_kwargs == {"enable_thinking": False}
