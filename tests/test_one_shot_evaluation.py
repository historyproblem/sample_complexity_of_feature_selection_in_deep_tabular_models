"""Production test accepts only frozen, fully trained one-shot deployments."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_one_shot_pruning as evaluation
from net_complexity.training.pruning_measurement import deployment_cost, mask_hash, state_hash


def accounting(kind="one_shot_reinit"):
    dense = kind == "dense_reference"
    return {"protocol": evaluation.PROTOCOL, "kind": kind, "status": "completed",
            "search_epochs_completed": 0 if dense else 150,
            "retrain_epochs_completed": 0 if dense else 150,
            "dense_epochs_completed": 150 if dense else 0,
            "global_epochs_completed": 150 if dense else 300,
            "total_epochs_allocated": 150 if dense else 300}


@pytest.mark.parametrize("kind", ["dense_reference", "one_shot_reinit"])
def test_exact_production_epoch_accounting(kind):
    assert evaluation.validate_epoch_accounting(accounting(kind)) == kind


@pytest.mark.parametrize("changes", [
    {"protocol": "adaptive_lambda_v1"}, {"status": "running"}, {"kind": "search"},
    {"search_epochs_completed": 149}, {"retrain_epochs_completed": 149},
    {"search_epochs_completed": 150.0}, {"dense_epochs_completed": 150},
    {"total_epochs_allocated": 150}, {"global_epochs_completed": 150},
])
def test_reject_partial_smoke_and_mislabelled_budget(changes):
    with pytest.raises(ValueError):
        evaluation.validate_epoch_accounting({**accounting(), **changes})


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Sequential(nn.BatchNorm2d(3), nn.Flatten()),
                                      nn.Linear(3 * 32 * 32, 10))

    def forward(self, x, y):
        logits = self.backbone(x)
        return SimpleNamespace(logits=logits, ce_loss=nn.functional.cross_entropy(logits, y))


@pytest.fixture
def saved_job(tmp_path, monkeypatch):
    job = "O1_internal_gap05_1"
    source = tmp_path / job
    source.mkdir()
    model = TinyClassifier()
    mask = {"backbone.layer1.0.mid1_gumbel_layer": [0]}
    validation = {"accuracy": .9, "ce_loss": .4, "correct_count": 4500, "example_count": 5000}
    state = {**accounting(), "seed": 42, "reinit_seed": 4242,
             "accepted_mask": mask, "accepted_mask_hash": mask_hash(mask),
             "validation": validation, "common_init_hash": "original_zero_epoch_initializer",
             "scratch_init_hash": "fresh_compact_initializer", "split_indices_hash": "same_split",
             "search_weights_reused_for_scratch": False, "search_bn_reused_for_scratch": False,
             "search_optimizer_reused_for_scratch": False, "search_controller_reused_for_scratch": False,
             "gate_regularization_normalization": "initial_channels", "final_cost": deployment_cost(model)}
    checkpoint = {"protocol": evaluation.PROTOCOL, "kind": "one_shot_reinit", "model_state_dict": model.state_dict(),
                  "model_state_hash": state_hash(model.state_dict()), "pruning_mask": mask,
                  "mask_hash": mask_hash(mask), "validation": validation,
                  "common_init_hash": state["common_init_hash"], "scratch_init_hash": state["scratch_init_hash"],
                  "global_epochs_consumed": 300, "provenance": {"phase": "scratch", "selected_epoch": 100,
                      "trained_from_scratch": True, "search_state_reused": False}}
    config = OmegaConf.create({"seed": 42, "dataloaders": {"taskname": "CIFAR10", "include_test": False},
        "training_arguments": {"evaluate_test": False, "adaptive_lambda": {"enabled": True}},
        "one_shot_pruning": {"protocol": evaluation.PROTOCOL, "search_epochs": 150, "retrain_epochs": 150,
                             "mask_threshold": .5, "probability_source": "raw_logits", "reinit_seed": 4242},
        "model": {"lambda_coef": .001, "criterion": {"_target_": "torch.nn.CrossEntropyLoss"},
                  "backbone": {"_target_": "net_complexity.wrappers.ResNet50", "num_classes": 10,
                               "in_channels": 3, "stem_kernel_size": 3, "stem_stride": 1,
                               "stem_padding": 1, "use_maxpool": False,
                               "resnet_block": {"gate_internal_width": True, "gate_output": False,
                                                "regularization_normalization": "initial_channels"}}}})
    torch.save(checkpoint, source / "deployment.pt")
    OmegaConf.save(config, source / "resolved_config.yaml")
    state.update(deployment_sha256=evaluation.file_hash(source / "deployment.pt"),
                 resolved_config_sha256=evaluation.file_hash(source / "resolved_config.yaml"),
                 deployment_model_state_hash=checkpoint["model_state_hash"])
    (source / "one_shot_state.json").write_text(json.dumps(state))
    monkeypatch.setattr(evaluation, "build_structurally_pruned_model_from_config", lambda *_: TinyClassifier())
    return tmp_path, job, state, checkpoint


def test_prepare_deployment_records_search_and_scratch_separately(saved_job):
    root, job, state, checkpoint = saved_job
    model, record = evaluation.prepare_job(root, job)
    assert record["epochs_consumed"] == 300
    assert record["search_epochs"] == record["retrain_epochs"] == 150
    assert record["dense_epochs"] == 0
    assert state_hash(model.state_dict()) == checkpoint["model_state_hash"]
    assert record["physical_parameters"] == state["final_cost"]["physical_total_parameters"]


@pytest.mark.parametrize("field,value", [
    ("scratch_init_hash", None), ("scratch_init_hash", "original_zero_epoch_initializer"),
    ("gate_regularization_normalization", "enabled_channels"), ("reinit_seed", 42),
    ("accepted_mask_hash", "wrong"), ("search_weights_reused_for_scratch", True),
    ("search_bn_reused_for_scratch", True), ("search_optimizer_reused_for_scratch", True),
    ("search_controller_reused_for_scratch", True),
])
def test_prepare_refuses_inconsistent_provenance(saved_job, field, value):
    root, job, state, _ = saved_job
    state[field] = value
    (root / job / "one_shot_state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError):
        evaluation.prepare_job(root, job)


def test_prepare_refuses_checkpoint_mutation(saved_job):
    root, job, _, checkpoint = saved_job
    checkpoint["model_state_dict"]["backbone.1.weight"].add_(1)
    torch.save(checkpoint, root / job / "deployment.pt")
    with pytest.raises(ValueError, match="hash"):
        evaluation.prepare_job(root, job)


def test_check_only_does_not_load_data_or_write(saved_job, monkeypatch):
    root, job, _, _ = saved_job
    monkeypatch.setattr(evaluation, "build_test_loader", lambda *_: pytest.fail("Test must not load"))
    args = SimpleNamespace(run_dir=root, output=None, jobs=[job], reference_source=None, check_only=True)
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
    evaluation.run(args)
    assert before == sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def test_bad_training_refused_before_test_access(saved_job, monkeypatch):
    root, job, state, _ = saved_job
    state["status"] = "failed"
    (root / job / "one_shot_state.json").write_text(json.dumps(state))
    monkeypatch.setattr(evaluation, "build_test_loader", lambda *_: pytest.fail("Test must not load"))
    args = SimpleNamespace(run_dir=root, output=None, jobs=[job], reference_source=None, check_only=False, device="cpu")
    with pytest.raises(ValueError, match="unfinished"):
        evaluation.run(args)
    assert not (root / "test_evaluation").exists()


def test_single_job_inference_checks_explicit_dense_reference(saved_job, monkeypatch):
    root, job, _, _ = saved_job
    original = evaluation.prepare_job

    def prepare(run_dir, name, source):
        if name == evaluation.DENSE:
            _, record = original(run_dir, job)
            return None, {**record, "common_init_hash": "different_reference_initializer"}
        return original(run_dir, name)

    monkeypatch.setattr(evaluation, "prepare_job", prepare)
    monkeypatch.setattr(evaluation, "build_test_loader", lambda *_: pytest.fail("Test must not load"))
    args = SimpleNamespace(run_dir=root, output=None, jobs=[job], reference_source=root / "dense",
                           check_only=False, device="cpu")
    with pytest.raises(ValueError, match="Unmatched common_init_hash"):
        evaluation.run(args)
    assert not (root / "test_evaluation").exists()


def test_comparison_uses_test_and_exposes_300_epoch_cost(saved_job):
    root, job, _, _ = saved_job
    _, record = evaluation.prepare_job(root, job)
    record["test"] = {"accuracy": .8953, "correct_count": 8953, "example_count": 10000, "ce_loss": .42}
    evaluation.write_comparison(root, [record])
    with (root / "test_comparison.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row["test_accuracy"]) == .8953
    assert float(row["validation_accuracy"]) == .9
    assert int(row["epochs_consumed"]) == 300
    assert int(row["search_epochs"]) == int(row["retrain_epochs"]) == 150
    assert "150 search + 150 fresh scratch" in (root / "test_comparison.md").read_text()


def test_frozen_inference_retains_weights_and_bn():
    torch.set_num_threads(1)
    model = TinyClassifier()
    data = torch.utils.data.TensorDataset(torch.rand(5, 3, 32, 32), torch.tensor([0, 1, 2, 3, 4]))
    before = state_hash(model.state_dict())
    metrics, arrays = evaluation.evaluate_fixed(model, torch.utils.data.DataLoader(data, batch_size=2),
                                                "cpu", job="synthetic", expected_examples=5)
    assert metrics["example_count"] == 5
    assert metrics["accuracy"] == metrics["correct_count"] / 5
    assert np.array_equal(arrays["index"], np.arange(5))
    assert state_hash(model.state_dict()) == before
