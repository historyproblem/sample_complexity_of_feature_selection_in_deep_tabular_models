import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_one_shot_pruning_test as evaluation
from net_complexity.training.one_shot_pruning import physical_architecture_signature
from net_complexity.training.one_shot_pruning_config import compose_config
from net_complexity.training.pruning_measurement import mask_hash, state_hash, write_json


class FrozenProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10, eps=0)

    def forward(self, x, y):
        assert not self.training
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        return SimpleNamespace(logits=self.bn(x))


@pytest.fixture
def search_export(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = compose_config()
    OmegaConf.save(config, tmp_path / "resolved_config.yaml", resolve=True)

    selected = {"epoch": 51, "model_state_dict": {"weight": torch.tensor([1.0])}}
    selected["model_state_hash"] = state_hash(selected["model_state_dict"])
    torch.save(selected, tmp_path / "selected_checkpoint.pt")
    identity = {
        "selected_checkpoint_id": "shared_search:epoch_0051:fixture",
        "selected_checkpoint_hash": evaluation.file_hash(tmp_path / "selected_checkpoint.pt"),
        "selected_model_state_hash": selected["model_state_hash"],
    }
    mask = {"backbone.layer1.0.mid1_gumbel_layer": [0]}
    identity["mask_hash"] = mask_hash(mask)
    ledger = {"global_training_epoch": 60, "search_epochs_consumed": 60,
              "optimizer_updates": 600, "consumed_training_examples": 2400}
    selection = {
        **identity,
        "pruning_mask": mask,
        "selected_epoch": 51,
        "quality_threshold": 0.85,
        "reference_epoch": 60,
        "policy": "best_feasible_compact",
        "trace": {},
        "search_ledger_consumed": ledger,
    }
    write_json(tmp_path / "selection.json", selection)

    widths = {
        f"backbone.layer{stage}.{index}.mid{mid}_gumbel_layer": 64 * 2 ** (stage - 1)
        for stage, blocks in enumerate((3, 4, 6, 3), 1)
        for index in range(blocks)
        for mid in (1, 2)
    }
    normalization = {
        "version": 1,
        "M0": len(widths),
        "n_b0": widths,
        "normalization": {name: "initial_channels" for name in widths},
        "scaling_contract": "survivor_equivalent_v1",
    }
    model = FrozenProbe()
    weights = model.state_dict()
    weights_hash = state_hash(weights)
    architecture_hash = hashlib.sha256(
        json.dumps(physical_architecture_signature(model), sort_keys=True).encode()
    ).hexdigest()
    cost = {"physical_total_parameters": 20, "conv_linear_macs_per_image": 123}
    diagnostics = {
        **identity,
        "pruning_mask": mask,
        "architecture_hash": architecture_hash,
        "status": "measured",
        "training_epochs": 0,
        "bn_calibration_batches": 0,
        "physical_validation": {"accuracy": 0.9, "ce_loss": 0.2},
        "physical_state_hash": weights_hash,
        "physical_cost": cost,
        "weights_and_bn_unchanged": True,
    }
    export = tmp_path / "export_only"
    export.mkdir()
    write_json(export / "diagnostics.json", diagnostics)
    torch.save({
        **identity,
        "protocol": str(config.one_shot.protocol),
        "artifact_type": "physical_ungated",
        "model_state_dict": weights,
        "model_state_hash": weights_hash,
        "pruning_mask": mask,
        "training_epochs": 0,
        "bn_calibration_batches": 0,
        "normalization_metadata": normalization,
    }, export / "deployment.pt")
    state = {
        "protocol": str(config.one_shot.protocol),
        "status": "search_only_completed",
        "search_only": True,
        "test_evaluated": False,
        "provenance": {
            "normalization": normalization,
            "inputs": {"initializer_model_state_hash": "initializer_hash"},
        },
        "shared_search_ledger": ledger,
        "selection": selection,
        "export_only": diagnostics,
        "branches": {},
    }
    write_json(tmp_path / "one_shot_state.json", state)
    monkeypatch.setattr(
        evaluation, "build_structurally_pruned_model_from_config", lambda *args: FrozenProbe()
    )
    monkeypatch.setattr(evaluation, "deployment_cost", lambda *args: cost)
    return tmp_path


def args(root, **overrides):
    values = dict(
        run_dir=root,
        output=None,
        check_only=False,
        search_export=True,
        device="cpu",
        data=root / "data",
        batch_size=128,
        num_workers=0,
        download=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_search_export_is_validated_without_claiming_complete_150_epochs(search_export):
    model, record = evaluation.prepare_search_export(search_export)
    assert not model.training
    assert record["search_epochs_consumed"] == 60
    assert record["selected_checkpoint_epoch"] == 51
    assert record["recovery_epochs_executed"] == 0
    assert record["per_model_150_epoch_protocol_complete"] is False
    assert record["artifact_type"] == "physical_ungated_search_export"


def test_search_export_frozen_test_is_quick_inference_only(search_export, monkeypatch, capsys):
    data = torch.zeros(10000, 10)
    data[:9000, 0] = 4
    data[9000:, 1] = 4
    dataset = TensorDataset(data, torch.zeros(10000, dtype=torch.long))
    dataset.targets = [0] * 10000
    loader = DataLoader(dataset, batch_size=128)
    monkeypatch.setattr(evaluation.frozen, "build_test_loader", lambda *args: (
        loader,
        {"dataset": "CIFAR10", "split": "official_test", "example_count": 10000,
         "ordered_data_and_labels_sha256": "synthetic_test_fixture"},
    ))

    before = {
        path: evaluation.file_hash(path)
        for path in search_export.rglob("*") if path.is_file()
    }
    report = evaluation.run(args(search_export))
    assert report["status"] == "completed"
    assert report["comparison_scope"] == "exploratory_search_only_incomplete_150_epoch_protocol"
    assert report["training_performed"] is False
    assert report["bn_recalibration"] is False
    assert report["test_based_selection"] is False
    assert report["runs"][0]["test"]["accuracy"] == pytest.approx(0.9)
    assert before == {path: evaluation.file_hash(path) for path in before}
    assert "[exploratory-search-test] search_export: accuracy=90.00%" in capsys.readouterr().out
    assert (search_export / "one_shot_test_evaluation/test_summary.json").is_file()


@pytest.mark.parametrize("mutation", ["running", "branch", "mask", "weights", "ledger"])
def test_invalid_search_export_is_refused_before_test_access(
    search_export, monkeypatch, mutation
):
    state_path = search_export / "one_shot_state.json"
    deployment_path = search_export / "export_only/deployment.pt"
    state = evaluation.read_json(state_path)
    deployment = torch.load(deployment_path, map_location="cpu", weights_only=True)
    if mutation == "running":
        state["status"] = "running"
    elif mutation == "branch":
        state["branches"] = {"unexpected": {}}
    elif mutation == "mask":
        deployment["pruning_mask"] = {}
    elif mutation == "weights":
        deployment["model_state_dict"]["bn.weight"][0] += 1
    elif mutation == "ledger":
        state["shared_search_ledger"]["global_training_epoch"] = 59
        state["selection"]["search_ledger_consumed"]["global_training_epoch"] = 59
        write_json(search_export / "selection.json", state["selection"])
    write_json(state_path, state)
    torch.save(deployment, deployment_path)
    monkeypatch.setattr(
        evaluation.frozen, "build_test_loader", lambda *args: pytest.fail("official test accessed")
    )
    with pytest.raises(ValueError):
        evaluation.run(args(search_export))
    assert not (search_export / "one_shot_test_evaluation").exists()


def test_search_export_check_only_does_not_open_test_data(search_export, monkeypatch):
    monkeypatch.setattr(
        evaluation.frozen, "build_test_loader", lambda *args: pytest.fail("official test accessed")
    )
    assert evaluation.run(args(search_export, check_only=True, device="cuda:0")) is None
    assert not (search_export / "one_shot_test_evaluation").exists()
