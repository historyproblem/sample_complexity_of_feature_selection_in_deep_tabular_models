"""Frozen per-deployment inference with synthetic probes, never real CIFAR data."""
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stochastic_pruning_test as testing
from net_complexity.training.one_shot_pruning_config import compose_config, ABLATION_PROTOCOL
from net_complexity.training.pruning_measurement import state_hash
from test_one_shot_pruning_evaluator import FrozenProbe


@pytest.fixture
def deployment(tmp_path):
    config = compose_config("experiment/pruning_v3/ablations_internal_stochastic_st")
    config.device = "cpu"
    model = FrozenProbe().eval()
    root = tmp_path / "profile" / "internal" / "recovery"
    root.mkdir(parents=True)
    OmegaConf.save(config, root / "resolved_config.yaml")
    branch = {"branch": "inherited", "status": "infeasible", "protocol": ABLATION_PROTOCOL,
              "ledger": {"global_training_epoch": 150, "search_epochs_consumed": 60},
              "final_training_epochs_executed": 90, "model_state_hash": state_hash(model.state_dict()),
              "mask_hash": "mask", "selected_final_epoch": 78, "quality_feasible": False,
              "validation": {"accuracy": .9, "ce_loss": .2},
              "final_cost": {"physical_total_parameters": 20, "conv_linear_macs_per_image": 0}}
    for name in ("inherited", "scratch", "random"):
        folder = root / name
        folder.mkdir()
        value = {**branch, "branch": name, "model_state_dict": model.state_dict()}
        torch.save(value, folder / "deployment.pt")
    return config, model, branch, root


def fake_loader(monkeypatch, reporter, calls):
    def build(*args):
        calls.append("loader")
        assert list((reporter.output / "frozen_deployments").glob("*.json"))
        # Consume each RNG deliberately, testing that later training sees no change.
        random.random(); np.random.random(); torch.rand(1)
        dataset = TensorDataset(torch.ones(10000, 10), torch.zeros(10000, dtype=torch.long))
        dataset.targets = np.zeros(10000, dtype=np.int64)
        return DataLoader(dataset, batch_size=1000), {
            "dataset": "CIFAR10", "split": "official_test", "example_count": 10000}
    monkeypatch.setattr(testing.evaluation.frozen, "build_test_loader", build)


def test_freeze_before_test_preserves_rng_bn_and_reuses_completed_results(tmp_path, monkeypatch, deployment):
    config, model, branch, root = deployment
    reporter = testing.DeploymentTests(tmp_path, ["profile"], "cpu")
    assert not reporter.output.exists()
    calls = []
    fake_loader(monkeypatch, reporter, calls)
    before = state_hash(model.state_dict())
    py, cpu, numpy = random.getstate(), torch.get_rng_state().clone(), np.random.get_state()
    for name in ("inherited", "scratch", "random"):
        record = reporter.evaluate(model, {**branch, "branch": name}, root, config)
        assert record["test"]["accuracy"] == 1 and record["test"]["example_count"] == 10000
        assert record["quality_feasible"] is False
    assert calls == ["loader"]
    assert state_hash(model.state_dict()) == before
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), cpu)
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert np.random.get_state()[2:] == numpy[2:]
    assert reporter.finish()["status"] == "completed"
    reloaded = testing.DeploymentTests(tmp_path, ["profile"], "cpu")
    monkeypatch.setattr(testing.evaluation.frozen, "build_test_loader", lambda *a: pytest.fail("completed test reran"))
    for name in ("inherited", "scratch", "random"):
        reloaded.evaluate(model, {**branch, "branch": name}, root, config)
    assert len(reloaded.finish()["runs"]) == 3


def test_test_cannot_start_before_full_150_epoch_budget(tmp_path, monkeypatch, deployment):
    config, model, branch, root = deployment
    reporter = testing.DeploymentTests(tmp_path, ["profile"])
    branch["ledger"]["global_training_epoch"] = 149
    monkeypatch.setattr(testing.evaluation.frozen, "build_test_loader", lambda *a: pytest.fail("test opened before budget complete"))
    with pytest.raises(ValueError, match="completed 60\\+90"):
        reporter.evaluate(model, branch, root, config)
    assert not reporter.output.exists()


def test_inference_failure_retains_frozen_fit_and_can_retry_without_training(tmp_path, monkeypatch, deployment):
    config, model, branch, root = deployment
    reporter = testing.DeploymentTests(tmp_path, ["profile"])
    calls = []
    fake_loader(monkeypatch, reporter, calls)
    original = testing.evaluation.frozen.evaluate_fixed
    monkeypatch.setattr(testing.evaluation.frozen, "evaluate_fixed", lambda *a, **kw: (_ for _ in ()).throw(OSError("transient test failure")))
    reporter.callback(config)(model, branch, root)
    assert reporter.report["errors"] and not reporter.report["runs"]
    checkpoint = root / "inherited/deployment.pt"
    before = testing.evaluation.file_hash(checkpoint)
    monkeypatch.setattr(testing.evaluation.frozen, "evaluate_fixed", original)
    record = reporter.evaluate(model, branch, root, config)
    assert record["test"]["example_count"] == 10000 and not reporter.report["errors"]
    assert testing.evaluation.file_hash(checkpoint) == before
