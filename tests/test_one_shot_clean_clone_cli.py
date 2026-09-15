"""Clean-clone opt-in command contracts; no real CIFAR training is launched."""
import builtins
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from net_complexity.training.one_shot_pruning_config import dense_source_paths

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_one_shot_pruning as launcher


@pytest.mark.parametrize("prepare_only", [False, True])
def test_new_reference_preview_needs_no_artifacts_runtime_cuda_or_data(
        tmp_path, monkeypatch, capsys, prepare_only):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried"))
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CUDA initialized"))
    from torchvision.datasets import CIFAR10
    monkeypatch.setattr(CIFAR10, "__init__", lambda *a, **kw: pytest.fail("CIFAR constructed"))
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert name not in {
            "net_complexity.training.one_shot_reference",
            "net_complexity.training.one_shot_pruning",
            "net_complexity.training.engine",
        }, f"training runtime imported: {name}"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    monkeypatch.setattr(launcher, "validate_inputs", lambda *a: pytest.fail("historical input validation"))
    args = (["--prepare-reference", "fresh reference"] if prepare_only else ["--from-scratch"])
    report = launcher.main(args + ["--dry-run"])
    assert json.loads(capsys.readouterr().out) == report
    assert report["inputs"]["status"] == "planned_new_reference"
    assert report["training_performed"] is False
    assert report["evaluate_test"] is False
    assert report["reference_preparation"]["dense_training_epochs"] == 150
    assert report["reference_preparation"]["initializer_trained_epochs"] == 0
    assert report["reference_preparation"]["dense_gates"] is False
    assert report["reference_preparation"]["dense_adaptive_lambda"] is False
    assert report["resolved_config"]["training_arguments"]["adaptive_lambda"]["enabled"] is True
    assert report["budget"]["per_branch_budget_including_shared_search"] == 150
    assert report["budget"]["total_unique_training_epochs_both_branches"] == 240
    assert report["budget"]["total_unique_training_epochs_including_reference"] == 390
    assert report["budget"]["total_unique_training_epochs_this_command"] == (150 if prepare_only else 390)
    assert set(report["execution_graph"]) == ({"dense_reference"} if prepare_only
                                               else {"dense_reference", "shared", "physical_branches"})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("args", [
    ["--from-scratch", "--dense-source", "old"],
    ["--from-scratch", "--prepare-reference", "new"],
    ["--prepare-reference", "new", "--dense-source", "old"],
    ["--reference-output", "new"],
    ["--prepare-reference", "new", "--reference-output", "other"],
    ["--prepare-reference", "new", "--output", "ignored"],
])
def test_conflicting_flags_rejected_without_composition(tmp_path, monkeypatch, args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "compose_config", lambda *a, **kw: pytest.fail("invalid flags composed"))
    with pytest.raises(SystemExit) as exc:
        launcher.main(args)
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["initializer.path", "reference.history_path", "reference.state_path", "reference.config_path"])
def test_new_reference_modes_refuse_input_path_overrides(tmp_path, monkeypatch, field):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--from-scratch", "--dry-run", "--override", f"accuracy_guided.{field}=old_input"])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("prepare_only", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_new_reference_root_keeps_all_generated_artifacts_inside_output(
        tmp_path, monkeypatch, prepare_only, dry_run):
    outside_initializer = tmp_path / "shared_random_seed42.pt"
    outside_initializer.write_bytes(b"preserve existing unrelated initializer")
    reference = tmp_path / "J1_dense_control"
    monkeypatch.setitem(sys.modules, "net_complexity.training.one_shot_reference", SimpleNamespace(
        prepare_dense_reference=lambda *a: pytest.fail("escaping output reached dense training")))
    args = (["--prepare-reference", str(reference)] if prepare_only else
            ["--from-scratch", "--reference-output", str(reference), "--output", str(tmp_path / "pruning")])
    with pytest.raises(SystemExit) as exc:
        launcher.main(args + (["--dry-run"] if dry_run else []))
    assert exc.value.code == 2
    assert outside_initializer.read_bytes() == b"preserve existing unrelated initializer"
    assert list(tmp_path.iterdir()) == [outside_initializer]


@pytest.mark.parametrize("existing", ["reference", "pruning"])
def test_existing_output_refused_before_reference_training(tmp_path, monkeypatch, existing):
    reference, pruning = tmp_path / "reference", tmp_path / "pruning"
    target = reference if existing == "reference" else pruning
    target.mkdir()
    monkeypatch.setitem(sys.modules, "net_complexity.training.one_shot_reference", SimpleNamespace(
        prepare_dense_reference=lambda *a: pytest.fail("existing output reached dense training")))
    with pytest.raises(FileExistsError, match="Refusing existing"):
        launcher.main(["--from-scratch", "--reference-output", str(reference), "--output", str(pruning)])
    assert list(target.iterdir()) == []
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("reference_name,pruning_name", [("same", "same"), ("root/ref", "root"), ("root", "root/pruning")])
def test_nested_outputs_rejected_before_creating_reference(tmp_path, monkeypatch, reference_name, pruning_name):
    monkeypatch.setitem(sys.modules, "net_complexity.training.one_shot_reference", SimpleNamespace(
        prepare_dense_reference=lambda *a: pytest.fail("nested outputs reached dense training")))
    with pytest.raises(ValueError, match="separate directories"):
        launcher.main(["--from-scratch", "--reference-output", str(tmp_path / reference_name),
                       "--output", str(tmp_path / pruning_name)])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("prepare_only", [False, True])
@pytest.mark.parametrize("pruning_status,pruning_epochs", [("completed", 240), ("no_feasible_search", 60)])
def test_reference_preparation_precedes_search_with_exact_generated_paths(
        tmp_path, monkeypatch, prepare_only, pruning_status, pruning_epochs):
    reference, pruning = tmp_path / "new reference", tmp_path / "pruning"
    calls = []
    generated_paths = dense_source_paths(reference)
    reference_state = {"status": "completed", "reference_training_epochs_actually_executed": 150,
                       "reference_origin": "new_shared_seed42_initializer", "common_init_hash": "common-state-hash",
                       "initializer_file_hash": "initializer-file-hash", "test_evaluated": False}
    pruning_state = {"status": pruning_status, "compute_ledger": {"actual_training_epochs_executed": pruning_epochs},
                     "per_branch_total_allocated": 150, "test_evaluated": False}

    def check_paths(config):
        actual = {**dict(config.accuracy_guided.reference),
                  "initializer_path": config.accuracy_guided.initializer.path}
        assert actual == generated_paths
        assert config.one_shot.search_epochs == 60
        assert config.one_shot.final_epochs == 90

    def prepare(config, output):
        check_paths(config)
        assert output == reference
        calls.append("prepare")
        reference.mkdir()
        return reference_state

    def preflight(config):
        check_paths(config)
        assert reference.exists()
        calls.append("validate")

    def run(config, output):
        check_paths(config)
        assert output == pruning
        calls.append("search_and_branches")
        output.mkdir()
        return pruning_state

    monkeypatch.setitem(sys.modules, "net_complexity.training.one_shot_reference", SimpleNamespace(prepare_dense_reference=prepare))
    monkeypatch.setitem(sys.modules, "net_complexity.training.one_shot_pruning", SimpleNamespace(run_one_shot_pruning=run))
    monkeypatch.setattr(launcher, "validate_inputs", preflight)
    args = (["--prepare-reference", str(reference)] if prepare_only else
            ["--from-scratch", "--reference-output", str(reference), "--output", str(pruning)])
    result = launcher.main(args)
    assert calls == (["prepare"] if prepare_only else ["prepare", "validate", "search_and_branches"])
    assert result is (reference_state if prepare_only else pruning_state)
    if prepare_only:
        assert not pruning.exists()
    else:
        manifest = json.loads((pruning / "clean_clone_state.json").read_text())
        assert manifest["status"] == pruning_status
        assert manifest["reference"]["root"] == str(reference)
        assert manifest["reference"]["state_path"] == generated_paths["state_path"]
        assert manifest["reference"]["common_init_hash"] == reference_state["common_init_hash"]
        assert manifest["reference"]["initializer_file_hash"] == reference_state["initializer_file_hash"]
        assert manifest["reference"]["status"] == "completed"
        assert manifest["pruning"]["root"] == str(pruning)
        assert manifest["compute_ledger"] == {
            "reference_training_epochs_actually_executed": 150,
            "pruning_training_epochs_actually_executed": pruning_epochs,
            "actual_training_epochs_executed": 150 + pruning_epochs,
        }
        assert manifest["per_pruning_branch_total_allocated"] == 150
        assert manifest["reference_cost_is_external_to_pruning_branch_budget"] is True
        assert manifest["test_evaluated"] is False


def test_failed_reference_never_reaches_search(tmp_path, monkeypatch):
    def prepare(*args):
        raise RuntimeError("reference preparation failed")

    monkeypatch.setitem(sys.modules, "net_complexity.training.one_shot_reference", SimpleNamespace(prepare_dense_reference=prepare))
    monkeypatch.setattr(launcher, "validate_inputs", lambda *a: pytest.fail("failed reference reached search preflight"))
    with pytest.raises(RuntimeError, match="reference preparation failed"):
        launcher.main(["--from-scratch", "--reference-output", str(tmp_path / "reference"),
                       "--output", str(tmp_path / "pruning")])
