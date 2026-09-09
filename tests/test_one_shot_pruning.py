from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from net_complexity.models.feature_selection import MaskedGumbelBottleneckLayer
from net_complexity.models.pruning_budget import gates
from net_complexity.training import one_shot_pruning as one
from net_complexity.training.pruning_measurement import state_hash


class TinyLoaders:
    """Real optimizer/BN/loader smoke, with a test-access sentinel."""
    def __init__(self, include_test=False, loader_seed=42, **kwargs):
        assert include_test is False
        rng = torch.Generator().manual_seed(2026)
        train = TensorDataset(torch.randn(2, 3, 8, 8, generator=rng), torch.tensor([0, 1]))
        valid = TensorDataset(torch.randn(3, 3, 8, 8, generator=rng), torch.tensor([0, 1, 2]))
        self.train_dataloader = DataLoader(train, batch_size=2, shuffle=True,
                                          generator=torch.Generator().manual_seed(loader_seed))
        self.valid_dataloader = DataLoader(valid, batch_size=2)

    @property
    def test_dataloader(self):
        raise AssertionError("Training must never access test")


def config(*, short=True, output=False):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base=None):
        cfg = compose(config_name=None, overrides=["+experiment=one_shot/O1_internal_gap05_1"])
    cfg.device = "cpu"
    cfg.dataloaders = {"_target_": "test_one_shot_pruning.TinyLoaders", "include_test": False,
                       "loader_seed": 42, "seed": 42}
    cfg.model.backbone.resnet_block.gate_internal_width = not output
    cfg.model.backbone.resnet_block.gate_output = output
    if short:
        cfg.one_shot_pruning.search_epochs = 1
        cfg.one_shot_pruning.retrain_epochs = 1
    return cfg


@pytest.fixture(scope="module")
def dense_reference(tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("one-shot") / "dense"
    cfg = config()
    state = one.run_dense_reference(cfg, root, allow_short_run=True)
    assert state["status"] == "completed"
    assert state["global_epochs_completed"] == state["dense_epochs_completed"] == 1
    assert state["search_epochs_completed"] == state["retrain_epochs_completed"] == 0
    yield root
    torch.set_num_threads(previous)


def with_reference(cfg, root):
    cfg.one_shot_pruning.initial_checkpoint = str(root / "initializer.pt")
    cfg.one_shot_pruning.reference_history = str(root / "global_history.csv")
    return cfg


def tiny_carrier(output=False):
    model = nn.Module()
    model.block = MaskedGumbelBottleneckLayer(16, 4, gate_internal_width=not output, gate_output=output,
                                              regularization_normalization="initial_channels")
    for gate in gates(model).values():
        with torch.no_grad():
            gate.logits[:, 0] = -10
            gate.logits[:, 1] = 10
    return model


@pytest.mark.parametrize("output", [False, True])
def test_raw_mask_includes_exact_half_and_ignores_bias_without_budget(output):
    model = tiny_carrier(output)
    for gate in gates(model).values():
        with torch.no_grad():
            gate.logits[0] = 0  # Exactly p_open = 0.5 must be removed.
            gate.logits[1] = torch.tensor([1.0, 0.0])  # Inside recovery's reopen interval.
        gate.set_open_bias(100.0)
        assert gate.get_selection_probs()[1] > 0.5
    mask, decision = one.mask_from_raw_logits(model)
    assert all(indices == [0, 1] for indices in mask.values())
    assert decision["temporary_open_bias_excluded"]
    assert decision["parameter_budget"] is None
    assert decision["minimum_keep_ratio"] is None
    assert decision["collapsed_boundaries"] == []


def test_collapse_is_reported_not_rescued_to_a_floor():
    model = tiny_carrier()
    gate = next(iter(gates(model).values()))
    with torch.no_grad():
        gate.logits.zero_()
    mask, decision = one.mask_from_raw_logits(model)
    assert decision["status"] == "collapsed"
    assert decision["collapsed_boundaries"]
    assert len(next(iter(mask.values()))) == 4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_fail_closed(value):
    model = tiny_carrier()
    with torch.no_grad():
        next(iter(gates(model).values())).logits[0, 0] = value
    with pytest.raises(FloatingPointError, match="logits"):
        one.mask_from_raw_logits(model)


def test_no_op_is_explicit():
    mask, report = one.mask_from_raw_logits(tiny_carrier())
    assert mask == {}
    assert report["removed_gate_channels"] == 0


def test_production_budget_is_150_plus_150_not_legacy_150():
    cfg = config(short=False)
    assert one.validate_config(cfg) == 300
    cfg.one_shot_pruning.search_epochs = 149
    with pytest.raises(ValueError, match="150"):
        one.validate_config(cfg)
    assert one.validate_config(cfg, allow_short_run=True) == 299


@pytest.mark.parametrize("key,value", [
    ("one_shot_pruning.max_param_fraction", 0.18),
    ("one_shot_pruning.probability_source", "channel_history_average"),
    ("one_shot_pruning.selection_checkpoint", "last.pt"),
    ("training_arguments.adaptive_lambda.enabled", False),
    ("training_arguments.evaluate_test", True),
    ("training_arguments.adaptive_lambda.baseline_history_dir", "hidden_baseline"),
    ("cyclic_channel_pruning.enabled", True),
    ("model.backbone.resnet_block.regularization_normalization", "enabled_channels"),
])
def test_unsupported_protocol_substitutions_rejected(key, value):
    cfg = config(short=False)
    OmegaConf.update(cfg, key, value, force_add=True)
    with pytest.raises(ValueError):
        one.validate_config(cfg)


@pytest.mark.parametrize("output", [False, True])
def test_fresh_compact_is_deterministic_without_gate_bn_or_search_handoff(output):
    cfg = config(output=output)
    suffix = "gumbel_layer" if output else "mid1_gumbel_layer"
    mask = {"backbone.layer1.0." + suffix: [0, 2]}
    a = one._fresh_compact(cfg, mask, 123)
    b = one._fresh_compact(cfg, mask, 123)
    assert state_hash(a.state_dict()) == state_hash(b.state_dict())
    del b
    c = one._fresh_compact(cfg, mask, 456)
    assert state_hash(a.state_dict()) != state_hash(c.state_dict())
    assert not gates(a)
    one.assert_fresh_compact(a)
    bn = next(module for module in a.modules() if isinstance(module, nn.BatchNorm2d))
    bn.running_mean.add_(1)
    with pytest.raises(ValueError, match="fresh statistics"):
        one.assert_fresh_compact(a)


def test_phase_schedules_and_data_order_reset(tmp_path):
    cfg = config(short=False)
    for name in ("search", "scratch", "dense"):
        phase = one._phase_config(cfg, tmp_path, name, 150, {})
        assert phase.scheduler.T_max == phase.training_arguments.num_epochs == 150
        assert phase.training_arguments.global_epoch_offset == 0
        assert phase.training_arguments.adaptive_lambda.enabled == (name == "search")
        assert phase.dataloaders.loader_seed == (4242 if name == "scratch" else 42)
        assert phase.training_arguments.audit_data_seed == phase.dataloaders.loader_seed


def test_reference_validation_and_changed_settings(dense_reference):
    cfg = config()
    state = one.validate_dense_reference(dense_reference, cfg, allow_short_run=True)
    assert state["schedule"] == "single_continuous_cosine"
    cfg.training_arguments.adaptive_lambda.soft_drop = 0.02
    cfg.training_arguments.adaptive_lambda.hard_drop = 0.05
    cfg.model.backbone.resnet_block.gate_output = True
    cfg.model.backbone.resnet_block.gate_internal_width = False
    one.validate_dense_reference(dense_reference, cfg, allow_short_run=True)
    cfg.optimizer.lr *= 2
    with pytest.raises(ValueError, match="training_signature"):
        one.validate_dense_reference(dense_reference, cfg, allow_short_run=True)


@pytest.mark.parametrize("key,value,match", [
    ("common_init_hash", "changed", "initialization hash"),
    ("resolved_config_sha256", "changed", "configuration file changed"),
    ("deployment_sha256", "changed", "deployment file changed"),
    ("reference_history_sha256", "changed", "validation history changed"),
    ("optimizer_steps_total", 99, "history is not continuous"),
    ("accepted_mask", {"fake": [0]}, "metadata mismatch"),
])
def test_dense_reference_tampering_fails_closed(tmp_path, dense_reference, key, value, match):
    # References are immutable; only a copied manifest is changed in this test.
    state = json.loads((dense_reference / "one_shot_state.json").read_text())
    state[key] = value
    one.write_json(tmp_path / "one_shot_state.json", state)
    for filename in ("resolved_config.yaml", "initializer.pt", "global_history.csv", "deployment.pt"):
        (tmp_path / filename).symlink_to(dense_reference / filename)
    with pytest.raises(ValueError, match=match):
        one.validate_dense_reference(tmp_path, config(), allow_short_run=True)


@pytest.mark.parametrize("output", [False, True])
def test_real_cpu_search_prune_reinitialize_retrain(tmp_path, monkeypatch, dense_reference, output):
    # Force a known initial CLOSED channel, then let the real engine train it.
    # Carrier construction precedes the source hash, so all provenance remains honest.
    instantiate = one.instantiate
    def initialized(*args, **kwargs):
        result = instantiate(*args, **kwargs)
        if isinstance(result, nn.Module):
            for gate in gates(result).values():
                with torch.no_grad():
                    gate.logits[0] = torch.tensor([10.0, -10.0])
        return result
    monkeypatch.setattr(one, "instantiate", initialized)
    cfg = with_reference(config(output=output), dense_reference)
    root = tmp_path / "job"
    state = one.run_one_shot_pruning(cfg, root, allow_short_run=True)
    assert state["status"] == "completed"
    assert state["search_epochs_completed"] == state["retrain_epochs_completed"] == 1
    assert state["global_epochs_completed"] == state["total_epochs_allocated"] == 2
    assert state["optimizer_steps_total"] == 2
    assert state["physical_prune_events"] == 1
    assert state["compression_occurred"]
    assert state["accepted_mask"]
    assert all(indices == [0] for indices in state["accepted_mask"].values())
    assert state["final_cost"]["physical_total_parameters"] < state["initial_cost"]["physical_total_parameters"]
    assert state["validation"]["example_count"] == 3
    for stage in state["stages"]:
        assert stage["epochs_completed"] == stage["epochs_allocated"] == 1
        assert stage["data_epoch_offset"] == 0
        assert stage["adaptive_controller_reused"] is False
        assert stage["optimizer_state_reused"] is False
    scratch = torch.load(root / "scratch_initializer.pt", map_location="cpu", weights_only=True)
    deployment = torch.load(root / "deployment.pt", map_location="cpu", weights_only=True)
    assert scratch["trained_epochs"] == 0
    assert scratch["seed"] == 4242
    assert scratch["search_state_reused"] is False
    assert scratch["model_state_hash"] == state["scratch_init_hash"]
    assert state["scratch_init_hash"] != state["mask_selection"]["disposable_trained_state_hash"]
    assert deployment["model_state_hash"] != state["scratch_init_hash"]
    assert not any("gumbel_layer" in key for key in scratch["model_state_dict"])
    assert "optimizer_state_dict" not in scratch
    for key, tensor in scratch["model_state_dict"].items():
        if key.endswith("running_mean") or key.endswith("num_batches_tracked"):
            assert not torch.count_nonzero(tensor)
        if key.endswith("running_var"):
            assert torch.equal(tensor, torch.ones_like(tensor))
    with (root / "global_history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["global_epoch"] for r in rows] == ["1", "2"]
    assert [r["local_epoch"] for r in rows] == ["1", "1"]
    assert [r["stage"] for r in rows] == ["search", "scratch"]
    assert float(rows[0]["lambda_used"]) > 0
    assert float(rows[1]["lambda_used"]) == 0
    assert rows[1]["adaptive_lambda_action"] == "disabled_no_gates"


def test_missing_reference_records_failure_not_completed(tmp_path):
    cfg = config()
    cfg.one_shot_pruning.reference_history = str(tmp_path / "missing" / "global_history.csv")
    cfg.one_shot_pruning.initial_checkpoint = str(tmp_path / "missing" / "initializer.pt")
    with pytest.raises(FileNotFoundError):
        one.run_one_shot_pruning(cfg, tmp_path / "failed", allow_short_run=True)
    state = json.loads((tmp_path / "failed" / "one_shot_state.json").read_text())
    assert state["status"] == "failed"
    assert state["global_epochs_completed"] == 0
    assert not (tmp_path / "failed" / "deployment.pt").exists()


def test_real_collapse_saves_diagnostics_and_never_retrains(tmp_path, monkeypatch, dense_reference):
    instantiate = one.instantiate
    def closed(*args, **kwargs):
        model = instantiate(*args, **kwargs)
        if isinstance(model, nn.Module):
            for gate in gates(model).values():
                with torch.no_grad():
                    gate.logits[:, 0] = 10
                    gate.logits[:, 1] = -10
        return model
    monkeypatch.setattr(one, "instantiate", closed)
    cfg = with_reference(config(), dense_reference)
    root = tmp_path / "collapsed"
    with pytest.raises(ValueError, match="entire boundary"):
        one.run_one_shot_pruning(cfg, root, allow_short_run=True)
    state = json.loads((root / "one_shot_state.json").read_text())
    decision = json.loads((root / "mask_selection.json").read_text())
    assert state["status"] == "failed"
    assert state["global_epochs_completed"] == state["search_epochs_completed"] == 1
    assert state["retrain_epochs_completed"] == 0
    assert decision["status"] == "collapsed"
    assert len(decision["collapsed_boundaries"]) == 32
    assert not (root / "scratch_initializer.pt").exists()
    assert not (root / "deployment.pt").exists()


def test_interrupted_search_records_only_completed_epochs(tmp_path, monkeypatch, dense_reference):
    from net_complexity.training.interruption import TrainingInterrupted
    actual = one.run_training
    def interrupted(*args, **kwargs):
        actual(*args, **kwargs)
        raise TrainingInterrupted(epoch=2, batches=0)
    monkeypatch.setattr(one, "run_training", interrupted)
    cfg = with_reference(config(), dense_reference)
    root = tmp_path / "interrupted"
    with pytest.raises(TrainingInterrupted):
        one.run_one_shot_pruning(cfg, root, allow_short_run=True)
    state = json.loads((root / "one_shot_state.json").read_text())
    assert state["status"] == "interrupted"
    assert state["global_epochs_completed"] == state["search_epochs_completed"] == 1
    assert state["retrain_epochs_completed"] == 0
    assert state["stages"][0]["status"] == "interrupted"
    assert not (root / "deployment.pt").exists()


def test_existing_run_is_never_overwritten(tmp_path):
    root = tmp_path / "existing"
    root.mkdir()
    (root / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        one.run_dense_reference(config(), root, allow_short_run=True)
    assert (root / "user.txt").read_text() == "keep"
