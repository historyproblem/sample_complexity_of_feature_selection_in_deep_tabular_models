import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from net_complexity.models import channel_pruning
from net_complexity.training import cyclic_channel_pruning


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def search_config(*overrides):
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        return compose(
            config_name="mobilenet_onepass_thr05_gap8_12pp",
            overrides=[
                "training_arguments.adaptive_lambda.baseline_history_dir=/test/dense_reference",
                "+cyclic_channel_pruning.search_only=true",
                *overrides,
            ],
        )


@pytest.mark.parametrize(
    ("threshold", "zero_probs", "removed_channels"),
    [(0.5, [0.8, 0.3, 0.05], [0]), (0.9, [0.8, 0.3, 0.05], [0, 1]), (0.9, [0.01], [])],
    ids=["threshold05", "threshold09", "no_channels"],
)
def test_search_only_trains_once_and_saves_real_structural_architecture_counts(
    monkeypatch, tmp_path, threshold, zero_probs, removed_channels
):
    cfg = search_config(f"cyclic_channel_pruning.g_prob_threshold={threshold}")
    gate = "backbone.features.3.mid_gumbel_layer"
    expected_mask = {gate: removed_channels} if removed_channels else {}
    calls = []
    constructed = []

    def fake_run_training(stage_cfg, **kwargs):
        calls.append(stage_cfg)
        assert stage_cfg.training_arguments.num_epochs == 30
        assert stage_cfg.scheduler.T_max == 30
        assert stage_cfg.training_arguments.adaptive_lambda.enabled is True
        assert stage_cfg.training_arguments.adaptive_lambda.log_step_init == "auto"
        assert stage_cfg.training_arguments.adaptive_lambda.adaptive_log_step_enabled is False
        assert stage_cfg.channel_pruning.structural is False
        assert stage_cfg.model.criterion.label_smoothing == 0
        return {
            "num_epochs_executed": 30,
            "full_train_time_sec": 12.5,
            "last_valid_metrics": {
                f"valid_{gate}.channel_{index:03d}_zero_prob": prob
                for index, prob in enumerate(zero_probs)
            },
            "adaptive_lambda_baseline": {
                "history_path": "/test/dense_reference/history.csv",
                "full_train_time_sec": 200.0,
                "num_epochs_executed": 86,
            },
        }

    real_builder = channel_pruning.build_structurally_pruned_model_from_config

    def inspect_structural_builder(architecture_cfg, pruning_cfg):
        assert architecture_cfg.model.backbone.pretrained_weights is None
        assert pruning_cfg.structural is True
        model = real_builder(architecture_cfg, pruning_cfg)
        block = model.backbone.features[3]
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        constructed.append({
            "params": sum(parameter.numel() for parameter in model.parameters()),
            "expand_shape": tuple(block.branch.expand[0].weight.shape),
            "depthwise_groups": block.branch.depthwise[0].groups,
            "project_shape": tuple(block.branch.project.weight.shape),
        })

        def unexpected_forward(*args, **kwargs):
            pytest.fail("Architecture counting must not execute model inference")

        model.forward = unexpected_forward
        return model

    monkeypatch.setattr(cyclic_channel_pruning, "run_training", fake_run_training)
    monkeypatch.setattr(channel_pruning, "build_structurally_pruned_model_from_config", inspect_structural_builder)
    rng_state = torch.random.get_rng_state().clone()
    result = cyclic_channel_pruning.run_cyclic_channel_pruning_training(cfg, tmp_path)

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert len(calls) == 1
    assert cfg.model.backbone.pretrained_weights == "IMAGENET1K_V2"
    assert result["search_only"] is True
    assert result["num_cycles_completed"] == 1
    assert result["recovery_results"] == []
    assert result["final_result"] is None
    assert result["final_disabled_channels"] == expected_mask
    assert len(constructed) == 2
    dense, compact = constructed
    hidden_width = 144 - len(removed_channels)
    assert dense["params"] == 2480072
    assert compact["params"] == 2480072 - 61 * len(removed_channels)
    assert compact["expand_shape"] == (hidden_width, 24, 1, 1)
    assert compact["depthwise_groups"] == hidden_width
    assert compact["project_shape"] == (24, hidden_width, 1, 1)
    architecture = result["architecture"]
    assert architecture["dense_total_params"] == dense["params"]
    assert architecture["total_params"] == compact["params"]
    assert architecture["removed_params"] == 61 * len(removed_channels)
    assert architecture["removed_fraction"] == pytest.approx(61 * len(removed_channels) / 2480072)
    assert architecture["channels_removed"] == len(removed_channels)
    assert architecture["structural"] is True
    assert architecture["gate_free"] is True
    assert architecture["trained"] is False
    assert architecture["weights_exported"] is False
    assert architecture["purpose"] == "architecture_counts_only"
    assert architecture["channel_mask"] == expected_mask
    assert Path(architecture["mask_file"]) == (tmp_path / "channel_mask.json").resolve()
    assert Path(architecture["architecture_file"]) == (tmp_path / "architecture.json").resolve()
    assert json.loads((tmp_path / "channel_mask.json").read_text()) == expected_mask
    assert json.loads((tmp_path / "architecture.json").read_text()) == architecture
    assert json.loads((tmp_path / "cyclic_summary.json").read_text()) == result
    assert not list(tmp_path.glob("*.pt"))
    cost = result["training_cost"]
    assert cost["num_epochs_executed"] == 30
    assert cost["full_train_time_sec"] == 12.5
    assert cost["recovery_train_time_sec"] == 0
    assert cost["final_retrain_time_sec"] == 0
    assert cost["final_finetune_time_sec"] == 0
    assert cost["one_time_baseline_cost_sec"] == 200
    assert cost["one_time_baseline_epochs"] == 86
    assert cost["baseline_included_in_full_train_time"] is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (["cyclic_channel_pruning.max_cycles=2"], "requires max_cycles=1"),
        (["+cyclic_channel_pruning.audit_protocol=true"], "does not support audit_protocol"),
        (["+cyclic_channel_pruning.weight_handoff.enabled=true"], "does not support weight_handoff"),
    ],
)
def test_search_only_rejects_incompatible_execution_before_training(monkeypatch, tmp_path, overrides, message):
    cfg = search_config(*overrides)

    def unexpected_training(*args, **kwargs):
        pytest.fail("Invalid search-only configuration must fail before training")

    monkeypatch.setattr(cyclic_channel_pruning, "run_training", unexpected_training)
    with pytest.raises(ValueError, match=message):
        cyclic_channel_pruning.run_cyclic_channel_pruning_training(cfg, tmp_path)
