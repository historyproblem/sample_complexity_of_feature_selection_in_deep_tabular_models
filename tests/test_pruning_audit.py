from __future__ import annotations

import json
import signal
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from net_complexity.metrics.base import MultiLossMetric
from net_complexity.metrics.classification import Accuracy
from net_complexity.models.feature_selection import MaskedGumbelBottleneckLayer
from net_complexity.models.pruned_bottleneck import PrunedGumbelBottleneck
from net_complexity.models.pruning_budget import PhysicalBudget, gates, select_by_budget, validate_mask
from net_complexity.training.pruning_audit import selected_model


def _output(correct, count, loss=1.0, **kwargs):
    logits = torch.zeros(count, 2)
    logits[:correct, 0] = 1
    logits[correct:, 1] = 1
    return SimpleNamespace(logits=logits, ce_loss=torch.tensor(float(loss)),
                           loss=torch.tensor(float(loss)), regularization_loss=None, **kwargs)


def test_weighted_accuracy_and_reentrant_loss():
    accuracy, losses = Accuracy(return_counts=True), MultiLossMetric()
    for n_correct, n, loss in ((128, 128, 1), (0, 8, 5)):
        targets = torch.zeros(n, dtype=torch.long)
        output = _output(n_correct, n, loss)
        accuracy.update(None, output, targets)
        losses.update(None, output, targets)
        assert losses.compute() == losses.compute()
    assert accuracy.compute() == {"accuracy": 128 / 136, "correct_count": 128, "example_count": 136}
    assert losses.compute()["ce_loss"] == (128 + 8 * 5) / 136
    losses.update(None, _output(1, 1, 7, mean_p_open=torch.tensor(0.25)), torch.zeros(1))
    assert losses.compute()["mean_p_open"] == 0.25
    losses.reset()
    assert losses.compute() == {}
    accuracy.reset()
    with pytest.raises(ValueError):
        accuracy.compute()


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("removed", [[], [0, 3, 9]])
def test_scatter_matches_legacy_outputs_and_gradients(stride, removed):
    downsample = nn.Sequential(nn.Conv2d(16, 16, 1, stride=stride), nn.BatchNorm2d(16)) if stride == 2 else None
    model = PrunedGumbelBottleneck(16, 4, i_downsample=downsample, stride=stride,
                                   disabled_channels=removed, disabled_mid1_channels=[1],
                                   disabled_mid2_channels=[2]).eval()
    legacy = deepcopy(model)
    matrix = torch.zeros(16, len(model.active_indices))
    matrix[model.active_indices, torch.arange(len(model.active_indices))] = 1
    x = torch.randn(2, 16, 6, 6, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    out = model(x)
    residual = legacy.relu(legacy.batch_norm1(legacy.conv1(x_ref)))
    residual = legacy.relu(legacy.batch_norm2(legacy.conv2(residual)))
    residual = legacy.batch_norm3(legacy.conv3(residual))
    identity = legacy.i_downsample(x_ref) if legacy.i_downsample is not None else x_ref
    expected = legacy.relu(identity + torch.einsum("pn,bnhw->bphw", matrix, residual))
    torch.testing.assert_close(out, expected)
    out.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, x_ref.grad)
    for actual, reference in zip(model.parameters(), legacy.parameters()):
        torch.testing.assert_close(actual.grad, reference.grad)
    checkpoint = legacy.state_dict()
    checkpoint["active_selection"] = matrix
    model.load_state_dict(checkpoint, strict=True)
    assert "active_selection" not in model.state_dict()


def test_strict_topology_load_and_indices():
    block = PrunedGumbelBottleneck(16, 4)
    state = block.state_dict()
    state["active_indices"] = state["active_indices"].flip(0)
    with pytest.raises(RuntimeError, match="topology"):
        block.load_state_dict(state)
    for indices in ([0, 0], [-1], [16]):
        with pytest.raises(ValueError):
            PrunedGumbelBottleneck(16, 4, disabled_channels=indices)


def tiny_carrier(output=False):
    model = nn.Module()
    model.block = MaskedGumbelBottleneckLayer(16, 4, gate_internal_width=True, gate_output=output)
    return model


def test_internal_only_has_no_output_selector_or_regularizer_parameters():
    model = tiny_carrier()
    assert set(gates(model)) == {"block.mid1_gumbel_layer", "block.mid2_gumbel_layer"}
    assert not any("block.gumbel_layer" in key for key in model.state_dict())
    assert isinstance(model.block.gumbel_layer, nn.Identity)


@pytest.mark.parametrize("job,selector_names", [
    ("D1_dense_control", {"gumbel_layer"}),
    ("J2_output_fixed", {"gumbel_layer"}),
    ("D2_internal_fixed", {"mid1_gumbel_layer", "mid2_gumbel_layer"}),
])
def test_resnet50_channel_history_logs_real_gates(tmp_path, monkeypatch, job, selector_names):
    import csv
    import gzip
    import sys
    from pathlib import Path
    from hydra.utils import instantiate
    from net_complexity.training.pruning_audit import build_structural
    from net_complexity.training.run_history import RunHistory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from pruning_pilot_common import config_for

    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "unused")
    cfg = config_for(job)
    cfg.run_history.root_dir = str(tmp_path)
    cfg.run_history.use_hydra_output_dir = False
    assert cfg.run_history.log_channel_history
    history = RunHistory(cfg)
    model = instantiate(cfg.model)
    selectors = gates(model)
    first_name = next(iter(selectors))
    # Permanent masks must be represented in probabilities, with original indices.
    selectors[first_name].channel_mask[0] = 0
    count = history.log_channel_history(3, model)
    assert count == sum(gate.logits.shape[0] for gate in selectors.values())
    with gzip.open(history.channel_history_path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == count
    assert {row["layer_name"] for row in rows} == set(selectors)
    assert {name.rsplit(".", 1)[-1] for name in selectors} == selector_names
    assert {int(row["stage_index"]) for row in rows} == {1, 2, 3, 4}
    probabilities = {name: gate.get_selection_probs().detach().tolist() for name, gate in selectors.items()}
    logits = {name: gate.logits.detach().tolist() for name, gate in selectors.items()}
    for row in rows:
        name = row["layer_name"]
        channel = int(row["channel_index"])
        assert int(row["epoch"]) == 3
        assert float(row["selection_prob"]) == pytest.approx(probabilities[name][channel])
        assert float(row["logit_off"]) == pytest.approx(logits[name][channel][0])
        assert float(row["logit_on"]) == pytest.approx(logits[name][channel][1])
    masked_row = next(row for row in rows if row["layer_name"] == first_name and row["channel_index"] == "0")
    assert float(masked_row["selection_prob"]) == 0
    # Recovery uses the same history config, but structural models have no gates.
    structural = build_structural(cfg, model, {first_name: [0]})
    assert history.log_channel_history(4, structural) == 0


def test_joint_internal_cost_recomputed_and_physically_exact():
    model = tiny_carrier()
    for gate in gates(model).values():
        with torch.no_grad():
            gate.logits[:, 0] = -10
            gate.logits[:, 1] = 10
            gate.logits[0] = torch.tensor([10.0, -10.0])
    budget = PhysicalBudget(model, {})
    assert budget.marginal("block.mid1_gumbel_layer") == 55
    assert budget.marginal("block.mid2_gumbel_layer") == 55
    initial = budget.total()
    mask, report = select_by_budget(model, {}, 101.5 / initial, 0.5)
    assert mask == {"block.mid1_gumbel_layer": [0], "block.mid2_gumbel_layer": [0]}
    physical = PrunedGumbelBottleneck(16, 4, disabled_mid1_channels=[0], disabled_mid2_channels=[0])
    assert report["removed_params"] == 101  # NOT static 55 + 55
    assert report["params_after"] == sum(p.numel() for p in physical.parameters())
    second, report2 = select_by_budget(model, mask, 0.3, 0.5)
    assert report2["params_before"] == report["params_after"]
    assert report2["removed_params"] <= int(0.3 * report["params_after"])
    validate_mask(model, second, previous=mask, min_keep_ratio=0.5)


@pytest.mark.parametrize("mask", [
    {"unknown.mid1_gumbel_layer": [0]}, {"block.gumbel_layer": [0]},
    {"block.mid1_gumbel_layer": [-1]}, {"block.mid1_gumbel_layer": [4]},
    {"block.mid1_gumbel_layer": [True]}, {"block.mid1_gumbel_layer": [0, 0]},
    {"block.mid1_gumbel_layer": [0, 1, 2]},
])
def test_masks_fail_closed(mask):
    with pytest.raises(ValueError):
        validate_mask(tiny_carrier(), mask, min_keep_ratio=0.5)


def test_mask_cannot_reopen_and_budget_zero_is_identity():
    model = tiny_carrier()
    previous = {"block.mid1_gumbel_layer": [0]}
    with pytest.raises(ValueError, match="Nonmonotone"):
        validate_mask(model, {}, previous=previous)
    assert select_by_budget(model, previous, 0, 0.5)[0] == previous


def test_deterministic_ranking_and_floor():
    model = tiny_carrier()
    a, report = select_by_budget(model, {}, 0.9, 0.5, ranking="random", seed=9)
    b, _ = select_by_budget(model, {}, 0.9, 0.5, ranking="random", seed=9)
    assert a == b
    assert all(len(indices) == 2 for indices in a.values())
    assert report["unused_budget_params"] > 0
    mask, _ = select_by_budget(model, {}, 0.4, 0.5, ranking="random", seed=9)
    for gate in gates(model).values():
        with torch.no_grad():
            gate.logits.neg_()
    assert mask == select_by_budget(model, {}, 0.4, 0.5, ranking="random", seed=9)[0]


def test_selection_loads_best_not_last(tmp_path):
    cfg = OmegaConf.create({"model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 2}})
    model = nn.Linear(2, 2)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    with torch.no_grad():
        model.weight.fill_(3)
    torch.save({"epoch": 1, "model_state_dict": model.state_dict()}, checkpoints / "best.pt")
    with torch.no_grad():
        model.weight.fill_(9)
    torch.save({"epoch": 2, "model_state_dict": model.state_dict()}, checkpoints / "last.pt")
    result = {"run_dir": str(tmp_path), "best_epoch": 1, "last_valid_metrics": {"fake": 99}}
    loaded, payload, path = selected_model(cfg, result)
    assert loaded.weight.unique().item() == 3
    assert payload["epoch"] == 1
    result["best_epoch"] = 2
    with pytest.raises(ValueError, match="epoch"):
        selected_model(cfg, result)


def test_classic_loader_does_not_construct_test(tmp_path, monkeypatch):
    from net_complexity.data import dataloaders as d
    calls = []

    class Dataset:
        def __init__(self, root, train, transform, download):
            assert train, "test dataset constructed"
            calls.append(train)
        def __len__(self):
            return 20
        def __getitem__(self, index):
            return torch.zeros(3, 32, 32), 0

    monkeypatch.setattr(d.datasets, "CIFAR10", Dataset)
    data = d.ClassicCVDataloaders(str(tmp_path), taskname="CIFAR10", include_test=False,
                                 num_workers=0, batch_size=4, loader_seed=42)
    assert calls == [True, True]
    assert data.test_dataloader is None
    assert len(data.valid_dataloader.dataset) == 2


def test_engine_test_sentinel_and_interrupted_checkpoint(tmp_path, monkeypatch):
    from net_complexity.training import engine, interruption
    from net_complexity.training.meta import Metrics
    from net_complexity.training.run_history import RunHistory
    from net_complexity.data.dataloaders import Dataloaders

    class Data(Dataloaders):
        @property
        def test_dataloader(self):
            raise AssertionError("test loader accessed")
    class Metric:
        def compute(self):
            return {}
        def reset(self):
            pass
    model = nn.Linear(1, 1)
    monkeypatch.setattr(engine, "train_epoch", lambda *a, **k: {})
    monkeypatch.setattr(engine, "evaluate", lambda *a, **k: None)
    result = engine.train(model, torch.optim.SGD(model.parameters(), lr=0.1),
                          None, Data(), OmegaConf.create({"num_epochs": 1, "evaluate_test": False}),
                          Metrics(Metric(), Metric(), Metric()), "cpu")
    assert result["test_evaluation_disabled"]
    assert result["test_metrics"] == {}
    with interruption.cooperative_signals():
        signal.raise_signal(signal.SIGTERM)
        with pytest.raises(interruption.TrainingInterrupted):
            interruption.check_stop(epoch=2, batches=3)
    interruption.check_stop()


def test_checkpoint_tie_break_and_rng_safe_loading(tmp_path):
    from net_complexity.training.run_history import RunHistory
    cfg = OmegaConf.create({"run_history": {
        "root_dir": str(tmp_path), "monitor": "valid_accuracy", "mode": "max",
        "secondary_monitor": "valid_ce_loss", "use_hydra_output_dir": False,
    }})
    history = RunHistory(cfg)
    assert history.should_update_best(1, {"valid_accuracy": 0.8, "valid_ce_loss": 0.9})
    assert history.should_update_best(2, {"valid_accuracy": 0.8, "valid_ce_loss": 0.7})
    assert not history.should_update_best(3, {"valid_accuracy": 0.8, "valid_ce_loss": 1.1})
    model = nn.Linear(1, 1)
    path = history.save_checkpoint("last.pt", model, torch.optim.SGD(model.parameters(), lr=0.1),
                                  epoch=2, metrics={})
    state = torch.load(path, weights_only=True)
    assert set(state["rng_state"]) == {"python", "numpy", "torch", "cuda"}
    assert state["global_epoch"] == 2


@pytest.mark.parametrize("when", ["before", "after"])
def test_transaction_rolls_back_weights_and_mask_without_extra_epochs(tmp_path, monkeypatch, when):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from pruning_pilot_common import config_for, make_initializer
    from net_complexity.training import pruning_audit as audit
    from net_complexity.models.channel_pruning import build_structurally_pruned_model_from_config
    from hydra.utils import instantiate

    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", str(tmp_path / "init.pt"))
    cfg = config_for("J3_internal_fixed")
    make_initializer(cfg, tmp_path / "init.pt")
    cfg.device = "cpu"
    cfg.dataloaders = {"_target_": "smoke_pruning_pilot.SmokeDataloaders", "loader_seed": 42}
    cfg.cyclic_channel_pruning.max_cycles = 2
    cfg.cyclic_channel_pruning.gumbel_epochs = 1
    cfg.cyclic_channel_pruning.recovery_epochs = 1
    cfg.cyclic_channel_pruning.final_epochs = 1
    calls = []
    first_search_weights = None

    def fake_training(config, model_initializer, epoch_end_callback):
        nonlocal first_search_weights
        model = (build_structurally_pruned_model_from_config(config, config.channel_pruning)
                 if config.channel_pruning.structural else instantiate(config.model))
        model_initializer(model)
        name = str(config.run_history.run_name)
        if name == "rollback_finetune":
            torch.testing.assert_close(model.backbone.fc.bias, first_search_weights)
            assert OmegaConf.to_container(config.channel_pruning.mask) == {}
        epochs = int(config.training_arguments.num_epochs)
        calls.append((name, epochs))
        with torch.no_grad():
            model.backbone.fc.bias.add_(len(calls))
        if len(calls) == 1:
            first_search_weights = model.backbone.fc.bias.detach().clone()
        directory = tmp_path / name
        (directory / "checkpoints").mkdir(parents=True)
        torch.save({"epoch": epochs, "model_state_dict": model.state_dict()},
                   directory / "checkpoints" / "best.pt")
        for epoch in range(1, epochs + 1):
            epoch_end_callback(epoch, {"lr": 0.001}, {"valid_accuracy": 0.9, "valid_ce_loss": 0.1},
                               model, None, None)
        return {"run_dir": str(directory), "best_epoch": epochs, "best_metric_value": 0.9,
                "num_epochs_executed": epochs, "test_metrics": {}, "test_evaluation_disabled": True}

    values = iter([0.9, 0.5, 0.91] if when == "before" else [0.9, 0.9, 0.5, 0.91])
    def evaluate(*args, **kwargs):
        accuracy = next(values)
        return {"accuracy": accuracy, "ce_loss": 0.1, "correct_count": 9, "example_count": 10}
    monkeypatch.setattr(audit, "run_training", fake_training)
    monkeypatch.setattr(audit, "evaluate_deployment", evaluate)
    monkeypatch.setattr(audit, "calibrate_bn", lambda *a, **k: 0)
    monkeypatch.setattr(audit, "committed_equivalence", lambda *a, **k: None)
    result = audit.run_fixed_pruning_pilot(cfg, tmp_path / "pilot")
    assert result["global_epochs_completed"] == 4
    assert sum(n for _, n in calls) == 4
    assert calls[-1] == ("rollback_finetune", 3 if when == "before" else 2)
    assert result["decisions"][0]["status"] == f"rejected_{when}_recovery"
    assert result["accepted_mask"] == {}
    assert not result["parameter_target_met"]
    payload = torch.load(tmp_path / "pilot" / "deployment.pt", weights_only=True)
    assert payload["pruning_mask"] == {}
    assert payload["validation"]["accuracy"] == 0.91


def test_real_engine_saves_partial_checkpoint_on_stop(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from pruning_pilot_common import config_for
    from net_complexity.training import engine, interruption
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "unused")
    cfg = config_for("J1_dense_control")
    cfg.device = "cpu"
    cfg.dataloaders = {"_target_": "smoke_pruning_pilot.SmokeDataloaders", "loader_seed": 42}
    cfg.run_history.root_dir = str(tmp_path)
    cfg.run_history.use_hydra_output_dir = False
    original_step = torch.optim.AdamW.step
    def step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        signal.raise_signal(signal.SIGTERM)
        return result
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    with interruption.cooperative_signals(), pytest.raises(interruption.TrainingInterrupted):
        engine.run_training(cfg)
    checkpoint, = tmp_path.rglob("interrupted.pt")
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["extra_state"]["completed_epochs"] == 0
    assert saved["extra_state"]["batches_in_partial_epoch"] == 1
    assert "train_dataloader" in saved["rng_state"]
    assert saved["optimizer_state_dict"]["state"]
