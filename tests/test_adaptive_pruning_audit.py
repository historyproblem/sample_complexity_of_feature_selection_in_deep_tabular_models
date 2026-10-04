"""Real adaptive search/physical recovery handoff, with no CIFAR/test access."""
import csv
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

from omegaconf import OmegaConf
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pruning_pilot_common import config_for, make_initializer
from net_complexity.training.pruning_audit import (
    ADAPTIVE_PROTOCOL, load_adaptive_reference, run_adaptive_pruning_pilot,
    run_fixed_pruning_pilot, validate_config,
)
from net_complexity.training.pruning_measurement import state_hash


def make_config(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", str(tmp_path / "init.pt"))
    cfg = config_for("J3_internal_fixed")
    cfg.cyclic_channel_pruning.audit_protocol = ADAPTIVE_PROTOCOL
    OmegaConf.update(cfg, "cyclic_channel_pruning.adaptive_reference_history",
                     str(tmp_path / "reference.csv"), force_add=True)
    cfg.training_arguments.adaptive_lambda.enabled = True
    cfg.training_arguments.adaptive_lambda.baseline_history_dir = None
    cfg.metrics.valid_metrics.append(OmegaConf.create({
        "_target_": "net_complexity.metrics.gumbel.GumbelProbMetric", "log_channel_zero_probs": False}))
    return cfg


def test_adaptive_contract_requires_controller_and_total_budget(tmp_path, monkeypatch):
    cfg = make_config(tmp_path, monkeypatch)
    assert validate_config(cfg) == 150
    for key, value in (
        ("training_arguments.adaptive_lambda.enabled", False),
        ("training_arguments.evaluate_test", True),
        ("training_arguments.adaptive_lambda.baseline_history_dir", "would_train_hidden_baseline"),
        ("cyclic_channel_pruning.final_epochs", 61),
        ("model.lambda_coef", 0),
        ("model.lambda_coef", float("nan")),
        ("cyclic_channel_pruning.adaptive_reference_history", ""),
        ("model.backbone.resnet_block.regularization_normalization", "unknown"),
    ):
        changed = deepcopy(cfg)
        OmegaConf.update(changed, key, value, force_add=True)
        with pytest.raises(ValueError):
            validate_config(changed)
    with pytest.raises(ValueError, match="adaptive"):
        run_fixed_pruning_pilot(cfg, tmp_path / "wrong_entrypoint")
    with pytest.raises(ValueError, match="fixed"):
        run_adaptive_pruning_pilot(config_for("J3_internal_fixed"), tmp_path / "wrong_entrypoint")
    assert not (tmp_path / "wrong_entrypoint").exists()


@pytest.mark.parametrize("rows", ["1,0.9\n1,0.8\n", "1,nan\n2,0.8\n", "1,0.9\n", "1,1.1\n2,0.8\n"])
def test_reference_must_be_complete_finite_and_unambiguous(tmp_path, rows):
    path = tmp_path / "ref.csv"
    path.write_text("global_epoch,valid_accuracy\n" + rows)
    with pytest.raises(ValueError):
        load_adaptive_reference(path, 2)


@pytest.mark.parametrize("normalization", ["enabled_channels", "initial_channels"])
def test_real_two_cycle_adaptive_pruning_keeps_selected_controller(tmp_path, monkeypatch, normalization):
    torch.set_num_threads(1)
    cfg = make_config(tmp_path, monkeypatch)
    OmegaConf.update(cfg, "model.backbone.resnet_block.regularization_normalization", normalization, force_add=True)
    cfg.device = "cpu"
    cfg.dataloaders = {"_target_": "smoke_pruning_pilot.SmokeDataloaders", "loader_seed": 42}
    c = cfg.cyclic_channel_pruning
    c.max_cycles, c.gumbel_epochs, c.recovery_epochs, c.final_epochs = 2, 3, 1, 1
    c.max_param_fraction = 0.03
    c.commit_guard.train_bn_calibration_batches = 1
    c.commit_guard.max_immediate_accuracy_drop = 1.0
    c.commit_guard.max_recovered_accuracy_drop = 1.0
    a = cfg.training_arguments.adaptive_lambda
    a.warmup_epochs, a.update_every_epochs, a.acc_window = 0, 1, 1
    # All observed accuracies >= the reference, so updates must actually occur.
    a.recovery.min_epoch = 0
    a.recovery.enabled = True
    a.adaptive_log_step_enabled = True
    (tmp_path / "reference.csv").write_text("global_epoch,valid_accuracy\n" +
        "".join(f"{epoch},0.0\n" for epoch in range(1, 9)))
    make_initializer(cfg, tmp_path / "init.pt")
    init_before = (tmp_path / "init.pt").read_bytes()
    result = run_adaptive_pruning_pilot(cfg, tmp_path / "run")
    assert result["status"] == "completed" and result["pilot_version"] == 2
    assert result["protocol"] == ADAPTIVE_PROTOCOL and result["adaptive_lambda_enabled"] is True
    assert result["gate_regularization_normalization"] == normalization
    assert result["global_epochs_completed"] == 8 and result["optimizer_steps_total"] == 16
    assert result["test_evaluated"] is False
    assert result["accepted_mask"] and all(d["status"] == "accepted" for d in result["decisions"])
    with (tmp_path / "run/global_history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["global_epoch"]) for r in rows] == list(range(1, 9))
    search = [r for r in rows if r["stage"].endswith("search")]
    recovery = [r for r in rows if r["stage"].endswith("recovery")]
    assert len(search) == 6 and len(recovery) == 2
    assert all(float(r["lambda_next"]) > float(r["lambda_used"]) for r in search)
    assert all(r["valid_average_zero_prob"] != "" for r in search)
    assert all(float(r["lambda_used"]) == 0 for r in recovery)
    assert all(r["adaptive_lambda_action"] == "held_no_structural_gates" for r in recovery)
    assert all(r["gate_regularization_normalization"] == normalization and r["job"] == "run" for r in rows)
    assert len({r["initial_gate_channels"] for r in rows}) == 1
    assert int(search[3]["remaining_gate_channels"]) < int(search[0]["remaining_gate_channels"])
    deployment = torch.load(tmp_path / "run/deployment.pt", weights_only=True)
    assert deployment["gate_regularization_normalization"] == normalization
    assert deployment["initial_gate_channels"] == result["initial_gate_channels"]
    handoffs = result["adaptive_controller_handoffs"]
    assert len(handoffs) == 2
    for stage, handoff in zip((s for s in result["stages"] if s["name"].endswith("search")), handoffs):
        best = torch.load(Path(stage["run_dir"]) / "checkpoints/best.pt", weights_only=True)
        assert best["extra_state"]["adaptive_lambda_state"] == handoff["controller_state"]
        assert handoff["selected_epoch"] == best["epoch"]
        assert handoff["controller_state"]["runtime"]["last_epoch"] == handoff["selected_global_epoch"]
    first_next = math.exp(handoffs[0]["controller_state"]["runtime"]["log_lambda"])
    assert float(search[3]["lambda_used"]) == pytest.approx(first_next)
    assert float(search[3]["lambda_used"]) != pytest.approx(float(cfg.model.lambda_coef))
    assert handoffs[1]["controller_state"]["runtime"]["last_epoch"] >= 5
    deployment = torch.load(tmp_path / "run/deployment.pt", weights_only=True)
    assert state_hash(deployment["model_state_dict"]) == deployment["model_state_hash"]
    assert (tmp_path / "init.pt").read_bytes() == init_before
    assert json.loads((tmp_path / "run/adaptive_controller.json").read_text())["stage"] == "cycle_1_search"
