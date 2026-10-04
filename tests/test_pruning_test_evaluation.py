import csv
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_pruning_test as evaluation
from net_complexity.training.pruning_measurement import mask_hash, state_hash, write_json


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10, eps=0)
        self.flags = []

    def forward(self, x, y):
        self.flags.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return SimpleNamespace(logits=self.bn(x))


def config():
    return OmegaConf.create({
        "seed": 42, "dataloaders": {"taskname": "CIFAR10"},
        "model": {"lambda_coef": 0.001, "criterion": {"_target_": "torch.nn.CrossEntropyLoss"},
                  "backbone": {"_target_": "net_complexity.wrappers.ResNet50", "num_classes": 10,
                               "in_channels": 3, "stem_kernel_size": 3, "stem_stride": 1,
                               "stem_padding": 1, "use_maxpool": False,
                               "resnet_block": {"gate_output": True, "gate_internal_width": True}}}})


def save_job(folder, model, mask, cost):
    folder.mkdir(parents=True)
    OmegaConf.save(config(), folder / "resolved_config.yaml")
    validation = {"accuracy": 0.94, "ce_loss": 0.4, "correct_count": 4700, "example_count": 5000}
    state = {
        "status": "completed", "pilot_version": 1, "total_epochs_allocated": 150,
        "global_epochs_completed": 150, "optimizer_steps_total": 52650, "seed": 42,
        "common_init_hash": "shared_init", "split_indices_hash": "shared_split",
        "accepted_mask": mask, "accepted_mask_hash": mask_hash(mask),
        "validation": validation, "final_cost": cost,
    }
    weights = model.state_dict()
    checkpoint = {
        "model_state_dict": weights, "model_state_hash": state_hash(weights),
        "pruning_mask": mask, "mask_hash": mask_hash(mask), "validation": validation,
        "common_init_hash": "shared_init", "global_epochs_consumed": 150,
        "provenance": {"name": "cycle_2_recovery", "best_epoch": 48},
    }
    write_json(folder / "pilot_state.json", state)
    torch.save(checkpoint, folder / "deployment.pt")


@pytest.fixture
def files(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "night"
    monkeypatch.setattr(evaluation, "build_structurally_pruned_model_from_config", lambda *a: Probe())
    monkeypatch.setattr(evaluation, "deployment_cost", lambda m: {
        "physical_total_parameters": sum(p.numel() for p in m.parameters()),
        "conv_linear_macs_per_image": 123})
    for job in evaluation.JOBS:
        mask = {} if job == evaluation.JOBS[0] else {"backbone.layer1.0.mid1_gumbel_layer": [0]}
        save_job(root / job, Probe(), mask, {
            "physical_total_parameters": 20, "conv_linear_macs_per_image": 123})
    return root


def arguments(root, **overrides):
    args = dict(run_dir=root, output=None, device="cpu", jobs=list(evaluation.JOBS),
                dense_source=None, data=root / "data", batch_size=128, num_workers=0,
                download=False, check_only=False)
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize("mutation", ["weights", "mask", "mask_hash", "validation", "initializer",
                                     "epochs", "parameters", "incomplete", "precision"])
def test_rejects_mismatched_deployment(files, mutation):
    folder = files / evaluation.JOBS[2]
    checkpoint = torch.load(folder / "deployment.pt", weights_only=True)
    state = evaluation.read_json(folder / "pilot_state.json")
    if mutation == "weights":
        checkpoint["model_state_dict"]["bn.weight"][0] += 1
    elif mutation == "mask":
        checkpoint["pruning_mask"] = {}
    elif mutation == "mask_hash":
        checkpoint["mask_hash"] = "bad"
    elif mutation == "validation":
        checkpoint["validation"]["accuracy"] = 0.95
    elif mutation == "initializer":
        checkpoint["common_init_hash"] = "different"
    elif mutation == "epochs":
        checkpoint["global_epochs_consumed"] = 151
    elif mutation == "parameters":
        state["final_cost"]["physical_total_parameters"] = 21
    elif mutation == "incomplete":
        state["status"] = "running"
    elif mutation == "precision":
        checkpoint["model_state_dict"]["bn.weight"] = checkpoint["model_state_dict"]["bn.weight"].double()
    torch.save(checkpoint, folder / "deployment.pt")
    write_json(folder / "pilot_state.json", state)
    with pytest.raises(ValueError):
        evaluation.prepare_job(files, evaluation.JOBS[2])


def test_prepares_frozen_model(files):
    model, record = evaluation.prepare_job(files, evaluation.JOBS[2])
    assert not model.training
    assert record["physical_parameters"] == 20
    assert record["selection_provenance"]["best_epoch"] == 48
    assert record["epochs_consumed"] == 150
    assert state_hash(model.state_dict()) == record["model_state_hash"]


@pytest.mark.parametrize("normalization", ["enabled_channels", "initial_channels"])
@pytest.mark.parametrize("bad", [None, "protocol", "disabled", "epochs", "normalization_state",
                                "normalization_checkpoint", "normalization_unknown"])
def test_adaptive_v2_deployment_requires_audited_config(files, bad, normalization):
    source = files / evaluation.JOBS[2]
    job = evaluation.ADAPTIVE_JOBS[0]
    target = files / job
    shutil.copytree(source, target)
    state = evaluation.read_json(target / "pilot_state.json")
    state.update(pilot_version=2, protocol="adaptive_lambda_v1", adaptive_lambda_enabled=True)
    checkpoint = torch.load(target / "deployment.pt", weights_only=True)
    config = OmegaConf.load(target / "resolved_config.yaml")
    OmegaConf.update(config, "cyclic_channel_pruning.audit_protocol", "adaptive_lambda_v1", force_add=True)
    OmegaConf.update(config, "training_arguments.adaptive_lambda.enabled", True, force_add=True)
    # Legacy artifacts omit normalization metadata; they must retain their
    # enabled-channel meaning. The new opt-in must be explicit everywhere.
    if normalization == "initial_channels":
        OmegaConf.update(config, "model.backbone.resnet_block.regularization_normalization",
                         normalization, force_add=True)
        state["gate_regularization_normalization"] = checkpoint["gate_regularization_normalization"] = normalization
        state["initial_gate_channels"] = {"backbone.layer1.0.mid1_gumbel_layer": 64}
        checkpoint["initial_gate_channels"] = dict(state["initial_gate_channels"])
    if bad == "protocol":
        state["protocol"] = "unknown"
    elif bad == "disabled":
        config.training_arguments.adaptive_lambda.enabled = False
    elif bad == "epochs":
        state["global_epochs_completed"] = state["total_epochs_allocated"] = 151
    elif bad == "normalization_state":
        state["gate_regularization_normalization"] = "initial_channels" if normalization == "enabled_channels" else "enabled_channels"
    elif bad == "normalization_checkpoint":
        checkpoint["gate_regularization_normalization"] = "initial_channels" if normalization == "enabled_channels" else "enabled_channels"
    elif bad == "normalization_unknown":
        OmegaConf.update(config, "model.backbone.resnet_block.regularization_normalization", "unknown", force_add=True)
    OmegaConf.save(config, target / "resolved_config.yaml")
    write_json(target / "pilot_state.json", state)
    torch.save(checkpoint, target / "deployment.pt")
    if bad:
        with pytest.raises(ValueError):
            evaluation.prepare_job(files, job)
    else:
        model, record = evaluation.prepare_job(files, job)
        assert not model.training and record["pilot_version"] == 2
        assert record["training_protocol"] == "adaptive_lambda_v1"
        assert record["gate_regularization_normalization"] == normalization


@pytest.mark.parametrize("bad_widths", [None, {}, {"layer": 0}, {"layer": True}, {"layer": 3.5}])
def test_initial_normalization_requires_original_width_provenance(files, bad_widths):
    source = files / evaluation.JOBS[2]
    state = evaluation.read_json(source / "pilot_state.json")
    checkpoint = torch.load(source / "deployment.pt", weights_only=True)
    cfg = OmegaConf.load(source / "resolved_config.yaml")
    state.update(pilot_version=2, protocol="adaptive_lambda_v1", adaptive_lambda_enabled=True,
                 gate_regularization_normalization="initial_channels", initial_gate_channels=bad_widths)
    checkpoint.update(gate_regularization_normalization="initial_channels", initial_gate_channels=bad_widths)
    OmegaConf.update(cfg, "cyclic_channel_pruning.audit_protocol", "adaptive_lambda_v1", force_add=True)
    OmegaConf.update(cfg, "training_arguments.adaptive_lambda.enabled", True, force_add=True)
    OmegaConf.update(cfg, "model.backbone.resnet_block.regularization_normalization", "initial_channels", force_add=True)
    OmegaConf.save(cfg, source / "resolved_config.yaml")
    write_json(source / "pilot_state.json", state)
    torch.save(checkpoint, source / "deployment.pt")
    with pytest.raises(ValueError, match="normalization widths"):
        evaluation.prepare_job(files, evaluation.JOBS[2])


@pytest.mark.parametrize("mode", ["recorded", "relocated", "explicit_parent", "explicit_job"])
def test_resolves_reused_dense(files, mode):
    job = evaluation.JOBS[0]
    source = files.parent / "dense_day" / job
    source.parent.mkdir()
    shutil.move(str(files / job), str(source))
    state = evaluation.read_json(source / "pilot_state.json")
    state["reused_from"] = str(source) if mode == "recorded" else f"/old_server/dense_day/{job}"
    write_json(files / job / "pilot_state.json", state)
    shutil.copyfile(source / "resolved_config.yaml", files / f"{job}_resolved.yaml")
    explicit = source.parent if mode == "explicit_parent" else source if mode == "explicit_job" else None
    _, record = evaluation.prepare_job(files, job, explicit)
    assert record["checkpoint"] == str(source / "deployment.pt")


@pytest.mark.parametrize("mask", [
    {"wrong": [0]}, {"backbone.layer4.3.mid1_gumbel_layer": [0]},
    {"backbone.layer1.0.mid1_gumbel_layer": [64]},
    {"backbone.layer1.0.mid1_gumbel_layer": [0, 0]},
    {"backbone.layer1.0.mid1_gumbel_layer": [True]},
])
def test_rejects_invalid_mask(mask):
    with pytest.raises(ValueError):
        evaluation.validate_config_and_mask(config(), mask)


def test_metrics_include_partial_batch_and_do_not_change_bn():
    x = torch.zeros(5, 10)
    x[:4, 0] = 4
    x[-1, 1] = 8
    y = torch.zeros(5, dtype=torch.long)
    model = Probe().train()
    before = state_hash(model.state_dict())
    metrics, arrays = evaluation.evaluate_fixed(
        model, DataLoader(TensorDataset(x, y), batch_size=2), "cpu", job="probe", expected_examples=5)
    assert metrics["accuracy"] == 4 / 5
    assert metrics["ce_loss"] == pytest.approx(torch.nn.functional.cross_entropy(x, y).item())
    np.testing.assert_array_equal(arrays["index"], np.arange(5))
    np.testing.assert_array_equal(arrays["label"], y.numpy())
    assert arrays["probabilities"].shape == (5, 10)
    assert model.flags == [(False, True)] * 3
    assert state_hash(model.state_dict()) == before
    assert model.bn.num_batches_tracked == 0
    assert not model.training


@pytest.mark.parametrize("bad", ["nan", "incomplete", "mutating_bn"])
def test_rejects_invalid_test_pass(bad):
    class BadProbe(Probe):
        def forward(self, x, y):
            output = super().forward(x, y)
            if bad == "nan":
                output.logits.fill_(float("nan"))
            if bad == "mutating_bn":
                self.bn.running_mean.add_(1)
            return output
    loader = DataLoader(TensorDataset(torch.zeros(3, 10), torch.zeros(3, dtype=torch.long)), batch_size=2)
    with pytest.raises((ValueError, FloatingPointError)):
        evaluation.evaluate_fixed(BadProbe(), loader, "cpu", job="probe",
                                  expected_examples=4 if bad == "incomplete" else 3)


def test_loader_uses_only_official_test(monkeypatch, tmp_path):
    calls = []

    class FakeDataset(Dataset):
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.data = np.zeros((10000, 1), dtype=np.uint8)
            self.targets = [0] * 10000
            self.classes = [str(i) for i in range(10)]

        def __len__(self):
            return 10000

    monkeypatch.setattr(evaluation, "CIFAR10", FakeDataset)
    loader, info = evaluation.build_test_loader(tmp_path, 128, 0, "cpu")
    assert len(calls) == 1
    assert calls[0]["train"] is False and calls[0]["download"] is False
    assert [type(t).__name__ for t in calls[0]["transform"].transforms] == ["ToTensor", "Normalize"]
    assert not loader.drop_last
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    assert len(loader) == 79
    assert info["example_count"] == 10000


def test_check_only_never_loads_test_or_writes(files, monkeypatch):
    monkeypatch.setattr(evaluation, "build_test_loader", lambda *a: pytest.fail("test accessed"))
    assert evaluation.run(arguments(files, check_only=True, device="cuda:0")) is None
    assert not (files / "test_evaluation").exists()


def test_all_checkpoints_are_validated_before_test(files, monkeypatch):
    (files / evaluation.JOBS[-1] / "deployment.pt").rename(files / "missing.pt")
    monkeypatch.setattr(evaluation, "build_test_loader", lambda *a: pytest.fail("test accessed"))
    with pytest.raises(FileNotFoundError):
        evaluation.run(arguments(files))
    assert not (files / "test_evaluation").exists()


def test_full_output_preserves_input_and_refuses_overwrite(files, monkeypatch):
    hashes = {p: evaluation.file_hash(p) for p in files.rglob("*") if p.is_file()}
    x = torch.zeros(10000, 10)
    x[:9000, 0] = 4
    x[9000:, 1] = 4
    dataset = TensorDataset(x, torch.zeros(10000, dtype=torch.long))
    dataset.targets = [0] * 10000
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    monkeypatch.setattr(evaluation, "build_test_loader", lambda *a: (loader, {"example_count": 10000}))
    report = evaluation.run(arguments(files))
    output = files / "test_evaluation"
    assert report["status"] == "completed"
    assert len(report["runs"]) == 4
    assert report["training_performed"] is False and report["bn_recalibration"] is False
    for row in report["runs"]:
        assert row["test"]["accuracy"] == 0.9
        with np.load(output / row["predictions"], allow_pickle=False) as arrays:
            assert arrays["index"][-1] == 9999 and arrays["prediction"][-1] == 1
            assert len(arrays["label"]) == 10000
    with (output / "test_comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and all(float(r["test_accuracy"]) == 0.9 for r in rows)
    assert evaluation.read_json(output / "evaluation_plan.json")["test_evaluated"] is False
    assert evaluation.read_json(output / "test_summary.json")["test_evaluated"] is True
    assert hashes == {p: evaluation.file_hash(p) for p in hashes}
    with pytest.raises(FileExistsError):
        evaluation.run(arguments(files))


@pytest.mark.parametrize("mask", [
    {}, {"backbone.layer1.0.gumbel_layer": [0, 2]},
    {"backbone.layer1.0.mid1_gumbel_layer": [0, 2], "backbone.layer4.2.mid2_gumbel_layer": [1, 3]},
])
def test_real_structural_checkpoint_round_trip(tmp_path, mask):
    torch.set_num_threads(1)
    cfg = config()
    cfg.model.lambda_coef = 0.0
    pruning = OmegaConf.create({"mode": "explicit", "structural": True, "enabled": True, "mask": mask})
    model = evaluation.build_structurally_pruned_model_from_config(cfg, pruning).eval()
    cost = evaluation.deployment_cost(model)
    save_job(tmp_path / evaluation.JOBS[2], model, mask, cost)
    restored, record = evaluation.prepare_job(tmp_path, evaluation.JOBS[2])
    x, y = torch.randn(2, 3, 32, 32), torch.tensor([0, 1])
    with torch.inference_mode():
        torch.testing.assert_close(model(x, y).logits, restored(x, y).logits)
    assert record["physical_parameters"] == cost["physical_total_parameters"]
