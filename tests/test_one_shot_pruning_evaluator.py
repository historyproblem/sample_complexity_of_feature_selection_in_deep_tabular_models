from copy import deepcopy
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
from net_complexity.training.one_shot_pruning_config import compose_config
from net_complexity.training.pruning_measurement import mask_hash, state_hash, write_json
from net_complexity.training.one_shot_pruning import physical_architecture_signature


class FrozenProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10, eps=0)

    def forward(self, x, y):
        assert not self.training and torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        return SimpleNamespace(logits=self.bn(x))


@pytest.fixture
def pair(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = compose_config()
    OmegaConf.save(config, tmp_path / "resolved_config.yaml")
    selected = {"epoch": 51, "model_state_dict": {"weight": torch.tensor([1.0])}}
    selected["model_state_hash"] = state_hash(selected["model_state_dict"])
    torch.save(selected, tmp_path / "selected_checkpoint.pt")
    identity = {"selected_checkpoint_id": "shared_search:51", "selected_model_state_hash": selected["model_state_hash"],
                "selected_checkpoint_hash": evaluation.file_hash(tmp_path / "selected_checkpoint.pt")}
    mask = {"backbone.layer1.0.mid1_gumbel_layer": [0]}
    selection = {**identity, "pruning_mask": mask, "mask_hash": mask_hash(mask),
                 "selected_epoch": 51, "quality_threshold": .85, "reference_epoch": 60,
                 "policy": "best_feasible_compact", "search_ledger_consumed": {
                     "global_training_epoch": 60, "search_epochs_consumed": 60,
                     "optimizer_updates": 600, "consumed_training_examples": 2400}}
    write_json(tmp_path / "selection.json", selection)
    widths = {f"backbone.layer{stage}.{index}.mid{mid}_gumbel_layer": 64 * 2 ** (stage - 1)
              for stage, blocks in enumerate((3, 4, 6, 3), 1) for index in range(blocks) for mid in (1, 2)}
    normalization = {"version": 1, "M0": len(widths), "n_b0": widths,
                     "normalization": {name: "initial_channels" for name in widths},
                     "scaling_contract": "survivor_equivalent_v1"}
    cost = {"physical_total_parameters": 20, "conv_linear_macs_per_image": 123}
    for branch in evaluation.BRANCHES:
        folder = tmp_path / branch
        folder.mkdir()
        model = FrozenProbe()
        if branch == "scratch":
            model.bn.bias.data[0] = .1
        weights = model.state_dict()
        architecture_hash = hashlib.sha256(json.dumps(physical_architecture_signature(model), sort_keys=True).encode()).hexdigest()
        state = {"protocol": evaluation.PROTOCOL, "artifact_type": "physical_ungated", "branch": branch,
            "status": "completed", **identity, "pruning_mask": mask, "mask_hash": mask_hash(mask),
            "architecture_hash": architecture_hash, "initialization_state_hash": f"initial_{branch}",
            "initialization": "selected_surviving_state" if branch == "inherited" else "pytorch_default_all_trainable_and_bn",
            "training_initializer_verified": True, "normalization_metadata": normalization,
            "model_state_hash": state_hash(weights), "quality_feasible": True,
            "final_training_epochs_executed": 90, "per_branch_total_allocated": 150,
            "selection_policy": "best_validation_accuracy", "reference_epoch": 150,
            "selected_final_epoch": 80, "quality_threshold": .85,
            "validation": {"accuracy": .9, "ce_loss": .2, "correct_count": 4500, "example_count": 5000},
            "ledger": {"global_training_epoch": 150, "search_epochs_consumed": 60,
                       "optimizer_updates": 1500, "consumed_training_examples": 6000},
            "provenance": {"split_indices_hash": "same_split", "seed": 42, "normalization": normalization,
                "resolved_config_sha256": hashlib.sha256(OmegaConf.to_yaml(config, resolve=True).encode()).hexdigest(),
                "source_weights": "shared_zero_epoch_initializer_only", "dense_reference_weights_loaded": False},
            "final_cost": cost}
        write_json(folder / "branch_state.json", state)
        torch.save({**state, "model_state_dict": weights}, folder / "deployment.pt")
    monkeypatch.setattr(evaluation, "build_structurally_pruned_model_from_config", lambda *a: FrozenProbe())
    monkeypatch.setattr(evaluation, "deployment_cost", lambda *a: cost)
    return tmp_path


def args(root, **overrides):
    values = dict(run_dir=root, output=None, check_only=False, device="cpu", data=root / "data",
                  batch_size=128, num_workers=0, download=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pair_is_frozen_same_architecture_and_selected_identity(pair):
    prepared = evaluation.prepare_branches(pair)
    assert [record["branch"] for _, record in prepared] == ["inherited", "scratch"]
    first, second = [record for _, record in prepared]
    assert first["selected_checkpoint_hash"] == second["selected_checkpoint_hash"]
    assert first["mask_hash"] == second["mask_hash"]
    assert first["ledger"] == second["ledger"]
    assert first["model_state_hash"] != second["model_state_hash"]
    for model, record in prepared:
        before = state_hash(model.state_dict())
        loader = DataLoader(TensorDataset(torch.zeros(5, 10), torch.zeros(5, dtype=torch.long)), batch_size=2)
        metrics, arrays = evaluation.frozen.evaluate_fixed(model, loader, "cpu", job=record["branch"], expected_examples=5)
        assert metrics["example_count"] == 5 and arrays["label"].shape == (5,)
        assert state_hash(model.state_dict()) == before and model.bn.num_batches_tracked.item() == 0


@pytest.mark.parametrize("mutation", ["incomplete", "wrong_branch", "gated", "gate_tensor", "weights", "mask",
    "selected_id", "selected_hash", "selected_model_hash", "ledger", "search_clock", "architecture", "normalization",
    "validation", "precision", "cost", "initialization", "selected_epoch", "selection_mask", "selection_reference",
    "missing_scratch", "smoke", "final_policy", "quality_status", "split_provenance"])
def test_invalid_pair_is_refused_before_test_access(pair, monkeypatch, mutation):
    folder = pair / "scratch"
    state = evaluation.read_json(folder / "branch_state.json")
    checkpoint = torch.load(folder / "deployment.pt", weights_only=True)
    selection = evaluation.read_json(pair / "selection.json")
    if mutation == "incomplete":
        state["status"] = checkpoint["status"] = "running"
    elif mutation == "wrong_branch":
        checkpoint["branch"] = "inherited"
    elif mutation == "gated":
        checkpoint["artifact_type"] = "gated_only"
    elif mutation == "gate_tensor":
        checkpoint["model_state_dict"]["gumbel_layer.logits"] = torch.ones(2)
    elif mutation == "weights":
        checkpoint["model_state_dict"]["bn.weight"][0] += 1
    elif mutation == "mask":
        state["pruning_mask"] = checkpoint["pruning_mask"] = {}
        state["mask_hash"] = checkpoint["mask_hash"] = mask_hash({})
    elif mutation.startswith("selected_") and mutation != "selected_epoch":
        key = {"selected_id": "selected_checkpoint_id", "selected_hash": "selected_checkpoint_hash",
               "selected_model_hash": "selected_model_state_hash"}[mutation]
        checkpoint[key] = "mismatch"
    elif mutation == "ledger":
        state["ledger"]["global_training_epoch"] = checkpoint["ledger"]["global_training_epoch"] = 149
    elif mutation == "search_clock":
        state["ledger"]["search_epochs_consumed"] = checkpoint["ledger"]["search_epochs_consumed"] = 61
    elif mutation == "architecture":
        state["architecture_hash"] = checkpoint["architecture_hash"] = "other_architecture"
    elif mutation == "normalization":
        state["normalization_metadata"]["M0"] = checkpoint["normalization_metadata"]["M0"] = 1
    elif mutation == "validation":
        checkpoint["validation"]["accuracy"] = .8
    elif mutation == "precision":
        checkpoint["model_state_dict"]["bn.weight"] = checkpoint["model_state_dict"]["bn.weight"].double()
    elif mutation == "cost":
        state["final_cost"]["physical_total_parameters"] += 1
    elif mutation == "initialization":
        state["training_initializer_verified"] = False
    elif mutation == "selected_epoch":
        selection["selected_epoch"] = 60
    elif mutation == "selection_mask":
        selection["pruning_mask"] = {}
    elif mutation == "selection_reference":
        selection["reference_epoch"] = 51
    elif mutation == "final_policy":
        state["selection_policy"] = checkpoint["selection_policy"] = "test_accuracy"
    elif mutation == "quality_status":
        state["quality_feasible"] = checkpoint["quality_feasible"] = False
    elif mutation == "split_provenance":
        state["provenance"]["split_indices_hash"] = checkpoint["provenance"]["split_indices_hash"] = "different_split"
    elif mutation == "smoke":
        config = OmegaConf.load(pair / "resolved_config.yaml")
        config.accuracy_guided.smoke = True
        OmegaConf.save(config, pair / "resolved_config.yaml")
    write_json(folder / "branch_state.json", state)
    torch.save(checkpoint, folder / "deployment.pt")
    write_json(pair / "selection.json", selection)
    if mutation == "missing_scratch":
        (folder / "deployment.pt").unlink()
    monkeypatch.setattr(evaluation.frozen, "build_test_loader", lambda *a: pytest.fail("official test accessed"))
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluation.run(args(pair))
    assert not (pair / "one_shot_test_evaluation").exists()


def test_check_only_uses_cpu_no_test_data_or_output(pair, monkeypatch):
    before = {path: evaluation.file_hash(path) for path in pair.rglob("*") if path.is_file()}
    monkeypatch.setattr(evaluation.frozen, "build_test_loader", lambda *a: pytest.fail("official test accessed"))
    assert evaluation.run(args(pair, check_only=True, device="cuda:0")) is None
    assert before == {path: evaluation.file_hash(path) for path in before}
    assert not (pair / "one_shot_test_evaluation").exists()


def test_finalized_infeasible_branch_keeps_honest_quality_status(pair):
    folder = pair / "scratch"
    state = evaluation.read_json(folder / "branch_state.json")
    checkpoint = torch.load(folder / "deployment.pt", weights_only=True)
    for artifact in (state, checkpoint):
        artifact["status"] = "infeasible"
        artifact["quality_feasible"] = False
        artifact["validation"].update(accuracy=.8, correct_count=4000)
    write_json(folder / "branch_state.json", state)
    torch.save(checkpoint, folder / "deployment.pt")
    prepared = evaluation.prepare_branches(pair)
    assert prepared[0][1]["quality_feasible"] is True
    assert prepared[1][1]["quality_feasible"] is False


def test_both_branches_use_one_loader_and_readonly_frozen_forward(pair, monkeypatch, capsys):
    before = {path: evaluation.file_hash(path) for path in pair.rglob("*") if path.is_file()}
    data = torch.zeros(10000, 10)
    data[:9000, 0] = 4
    data[9000:, 1] = 4
    dataset = TensorDataset(data, torch.zeros(10000, dtype=torch.long))
    dataset.targets = [0] * 10000
    loader = DataLoader(dataset, batch_size=128)
    calls = []

    def one_loader(*arguments):
        calls.append(arguments)
        return loader, {"dataset": "CIFAR10", "split": "official_test", "example_count": 10000,
                        "ordered_data_and_labels_sha256": "synthetic_test_fixture"}

    monkeypatch.setattr(evaluation.frozen, "build_test_loader", one_loader)
    report = evaluation.run(args(pair))
    assert len(calls) == 1 and report["status"] == "completed"
    assert len(report["runs"]) == 2 and report["planned_branches"] == ["inherited", "scratch"]
    assert report["training_performed"] is False and report["bn_recalibration"] is False
    assert report["test_based_selection"] is False
    for row in report["runs"]:
        assert row["test"]["accuracy"] == .9 and row["test"]["example_count"] == 10000
        assert row["model_state_unchanged"] and row["bn_counters_unchanged"]
        with np.load(pair / "one_shot_test_evaluation" / row["predictions"], allow_pickle=False) as arrays:
            assert arrays["index"][-1] == 9999 and arrays["prediction"][-1] == 1
    console = capsys.readouterr().out
    assert "[official-test] inherited: accuracy=90.00%" in console
    assert "[official-test] scratch: accuracy=90.00%" in console
    assert "inherited-minus-scratch=+0.00 pp" in console
    assert "test_summary.json" in console
    assert before == {path: evaluation.file_hash(path) for path in before}
    with pytest.raises(FileExistsError):
        evaluation.run(args(pair))


def test_checked_in_server_config_resolves_reusable_paths(pair):
    resolved = evaluation.evaluation_args_from_config(
        evaluation.DEFAULT_CONFIG, run_dir=pair, data=pair / "cached-data", device="cpu",
        output=pair / "configured-test", num_workers=0)
    assert resolved.run_dir == pair.resolve()
    assert resolved.data == (pair / "cached-data").resolve()
    assert resolved.output == (pair / "configured-test").resolve()
    assert resolved.device == "cpu"
    assert resolved.batch_size == 128 and resolved.num_workers == 0
    assert resolved.download is False and resolved.check_only is False


def test_checked_in_config_cli_supports_artifact_only_preflight(pair, monkeypatch, capsys):
    monkeypatch.setattr(evaluation.frozen, "build_test_loader", lambda *a: pytest.fail("official test accessed"))
    assert evaluation.main([
        "--config", str(evaluation.DEFAULT_CONFIG), "--run-dir", str(pair),
        "--device", "cpu", "--check-only",
    ]) is None
    assert "Official test data not loaded" in capsys.readouterr().out
    assert not (pair / "test_evaluation").exists()
