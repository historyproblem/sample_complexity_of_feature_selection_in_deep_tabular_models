import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_resnet50_cifar10_repro20", ROOT / "scripts/launch_resnet50_cifar10_repro20.py",
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

CHILD_SPEC = importlib.util.spec_from_file_location(
    "repro20_one_shot_cli", ROOT / "scripts/launch_one_shot_pruning.py",
)
child_launcher = importlib.util.module_from_spec(CHILD_SPEC)
CHILD_SPEC.loader.exec_module(child_launcher)


def test_cpu_recipe_preserves_full_search_horizon_and_automatic_lambda():
    report = launcher.validate_recipe(launcher.load_study())
    assert report["search_horizon_epochs"] == report["scheduler_T_max"] == 60
    assert report["search_epochs_per_repeat_this_command"] == 20
    assert report["total_training_epochs_this_command"] == 40
    assert report["planned_per_model_budget"] == 150
    assert report["initial_search_warmup"] == 10
    assert report["resolved_log_step"] == pytest.approx(0.9210340371976183)
    assert report["test_evaluation"] is report["checkpoint_selection"] is False


@pytest.mark.parametrize("key,value", [
    ("search_epochs", 20), ("repeats", 3), ("soft_drop", 0.0025),
    ("min_keep_ratio", 0.001), ("expected_initializer_sha256", "other"),
])
def test_manifest_cannot_silently_change_the_diagnostic_question(tmp_path, key, value):
    study = launcher.load_study()
    study[key] = value
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError, match=key):
        launcher.load_study(path)


def test_validate_only_never_checks_inputs_or_launches_children(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU recipe validation must not touch runtime inputs")

    monkeypatch.setattr(launcher, "validate_reference", forbidden)
    monkeypatch.setattr(launcher.epoch_split, "_run", forbidden)
    monkeypatch.setattr(launcher.epoch_split, "_environment_preflight", forbidden)
    output = tmp_path / "absent"
    result = launcher.main(["--validate-only", "--output", str(output)])
    assert result["total_training_epochs_this_command"] == 40
    assert not output.exists()


def test_fresh_clone_error_explains_original_input_transfer_before_gpu_or_training(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Missing reference must stop before GPU preflight or training")

    monkeypatch.setattr(launcher.epoch_split, "_run", forbidden)
    monkeypatch.setattr(launcher.epoch_split, "_environment_preflight", forbidden)
    output = tmp_path / "new_run"
    with pytest.raises(FileNotFoundError) as error:
        launcher.main(["--reference-output", str(tmp_path / "missing_reference"),
                       "--output", str(output)])
    message = str(error.value)
    for expected in ("global_history.csv", "pilot_state.json", "resolved_config.yaml",
                     "shared_random_seed42.pt", "resnet50_repro20_inputs.tar.gz",
                     "--reference-output PATH"):
        assert expected in message
    assert "--from-scratch" not in message and "--dense-source" not in message
    assert not output.exists()


def write_prefix(output, *, diverged=False):
    run_dir = output / "shared_search/training/fixture"
    run_dir.mkdir(parents=True)
    state = {
        "status": "search_paused", "search_only": True, "test_evaluated": False,
        "branches": {}, "per_branch_total_allocated": 150,
        "shared_search_ledger": {
            "global_training_epoch": 20, "search_epochs_consumed": 20, "optimizer_updates": 20,
        },
        "stages": {"shared_search": {
            "epochs_allocated": 60, "scheduler": {"T_max": 60},
            "training_initializer_verified": True, "initialization_state_hash": "same",
            "run_dir": str(run_dir),
        }},
    }
    (output / "one_shot_state.json").write_text(json.dumps(state))
    with (run_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "valid_accuracy", "lambda_used", "lr"])
        writer.writeheader()
        for epoch in range(1, 21):
            writer.writerow({"epoch": epoch, "valid_accuracy": 0.51 if diverged else 0.50,
                             "lambda_used": 0.001, "lr": 0.001})
    diagnostics = output / "search_diagnostics"
    diagnostics.mkdir()
    (diagnostics / "environment.json").write_text(json.dumps({"backend": "unchanged"}))
    events = []
    for epoch in range(1, 21):
        events.extend([
            {"kind": "batch_input", "epoch": epoch, "batch_index": 0, "sha256": "same_batch"},
            {"kind": "epoch_end", "epoch": epoch,
             "model_state_sha256": "different" if diverged else "same"},
        ])
    (diagnostics / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))


def test_two_fresh_identical_child_recipes_and_nonfatal_divergence(tmp_path, monkeypatch):
    output = tmp_path / "result"
    calls = []
    reference = {"root": str(tmp_path / "ref"), "sha256": {"history_path": "pinned"}}
    monkeypatch.setattr(launcher, "validate_reference", lambda *args: copy.deepcopy(reference))
    monkeypatch.setattr(launcher.epoch_split, "_environment_preflight", lambda *args: {})

    def run(command):
        destination = Path(command[command.index("--output") + 1])
        assert not destination.exists()
        if calls:
            assert (output / "repeat_1/one_shot_state.json").is_file()
        calls.append(command)
        write_prefix(destination, diverged=len(calls) == 2)

    monkeypatch.setattr(launcher.epoch_split, "_run", run)
    result = launcher.main(["--output", str(output), "--reference-output", str(tmp_path / "ref")])
    assert result["status"] == "completed"
    assert result["comparison"]["comparison"] == "diverged"
    assert len(calls) == 2
    first, second = copy.deepcopy(calls)
    first[first.index("--output") + 1] = "output"
    second[second.index("--output") + 1] = "output"
    assert first == second
    assert "--search-only" in first and "--skip-test" in first and "--search-diagnostics" in first
    assert first[first.index("--stop-after-search-epoch") + 1] == "20"
    assert not any("num_epochs=" in value or "T_max=" in value for value in first)
    assert result["comparison"]["first_history_difference"]["epoch"] == 1
    assert result["comparison"]["first_trace_difference"]["kind"] == "epoch_end"
    assert (output / "comparison.csv").is_file() and (output / "report.md").is_file()
    with pytest.raises(ValueError, match="existing output"):
        launcher.main(["--output", str(output)])
    assert len(calls) == 2


@pytest.mark.parametrize("bad_artifact", ["wrong_ledger", "shortened_scheduler", "selection", "missing_trace_epoch"])
def test_incomplete_or_accidentally_exported_prefix_is_rejected(tmp_path, bad_artifact):
    output = tmp_path / "repeat"
    write_prefix(output)
    state_path = output / "one_shot_state.json"
    state = json.loads(state_path.read_text())
    if bad_artifact == "wrong_ledger":
        state["shared_search_ledger"]["global_training_epoch"] = 19
    elif bad_artifact == "shortened_scheduler":
        state["stages"]["shared_search"]["scheduler"]["T_max"] = 20
    elif bad_artifact == "selection":
        (output / "selection.json").write_text("{}\n")
    else:
        trace_path = output / "search_diagnostics/trace.jsonl"
        trace_path.write_text("\n".join(trace_path.read_text().splitlines()[:-1]) + "\n")
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        launcher.read_repeat(output)


def test_wrong_pinned_reference_fails_before_any_child(tmp_path, monkeypatch):
    from net_complexity.training import one_shot_pruning_config as schema

    study = launcher.load_study()
    monkeypatch.setattr(schema, "validate_inputs", lambda config: {
        "status": "ready", "sha256": {"history_path": "wrong", "initializer_path": "wrong"},
    })
    calls = []
    monkeypatch.setattr(launcher.epoch_split, "_run", lambda command: calls.append(command))
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="Wrong existing reference"):
        launcher.main(["--output", str(output), "--reference-output", str(tmp_path / "reference")])
    assert calls == [] and not output.exists()


def test_exact_replay_reports_identical_without_inventing_divergence(tmp_path):
    repeats = []
    for identifier in ("repeat_1", "repeat_2"):
        output = tmp_path / identifier
        write_prefix(output)
        repeats.append(launcher.read_repeat(output))
    result = launcher.compare_repeats(repeats, tmp_path)
    assert result["comparison"] == "identical"
    assert result["first_history_difference"] is None
    assert result["first_trace_difference"] is None
    assert result["first_environment_difference"] is None


def test_missing_batch_hash_is_an_incomplete_artifact_not_identical_replay(tmp_path):
    write_prefix(tmp_path / "repeat")
    path = tmp_path / "repeat/search_diagnostics/trace.jsonl"
    path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(ValueError, match="consumed training batches"):
        launcher.read_repeat(tmp_path / "repeat")


def test_generated_child_command_resolves_real_dry_run_as_20_epoch_prefix(tmp_path):
    study = launcher.load_study()
    output = tmp_path / "repeat_1"
    command = launcher.repeat_command(study, tmp_path / "missing_reference", output)
    report = child_launcher.main(command[2:] + ["--dry-run"])
    assert report["mode"] == "search_prefix"
    assert report["one_shot"]["search_epochs"] == 60
    assert report["one_shot"]["final_epochs"] == 90
    assert report["budget"]["per_branch_budget_including_shared_search"] == 150
    assert report["budget"]["search_training_epochs_this_command"] == 20
    assert report["budget"]["total_unique_training_epochs_this_command"] == 20
    assert report["budget"]["recovery_training_epochs_this_command"] == 0
    assert report["requested_execution"] == {
        "search_only": True, "stops_after": "complete_search_epoch",
        "stop_after_search_epoch": 20, "search_diagnostics": True,
    }
    assert report["official_test_evaluation"]["automatic_after_frozen_deployments"] is False
    assert not output.exists()


@pytest.mark.parametrize("arguments", [
    ["--stop-after-search-epoch", "20"],
    ["--search-only", "--stop-after-search-epoch", "0"],
    ["--search-only", "--stop-after-search-epoch", "60"],
    ["--search-only", "--stop-after-search-epoch", "61"],
    ["--reuse-search", "prior", "--search-diagnostics"],
    ["--reuse-search", "prior", "--resume-search", "paused"],
    ["--search-only", "--stop-after-search-epoch", "20", "--from-scratch"],
])
def test_child_cli_rejects_unsafe_pause_or_reuse_combinations(arguments):
    with pytest.raises(SystemExit) as exc:
        child_launcher.main([
            "--config-name", launcher.load_study()["search_config"], "--dry-run", *arguments,
        ])
    assert exc.value.code == 2


def test_resume_preview_charges_only_remaining_search_and_cannot_rewind(tmp_path):
    paused = tmp_path / "paused"
    paused.mkdir()
    (paused / "one_shot_state.json").write_text(json.dumps({
        "status": "search_paused", "shared_search_ledger": {"global_training_epoch": 20},
    }))
    arguments = [
        "--config-name", launcher.load_study()["search_config"],
        "--dense-source", str(tmp_path / "missing_reference"),
        "--output", str(tmp_path / "continued"), "--resume-search", str(paused),
        "--search-only", "--skip-test", "--dry-run",
    ]
    report = child_launcher.main(arguments)
    assert report["mode"] == "resume_paused_search"
    assert report["one_shot"]["search_epochs"] == 60
    assert report["budget"]["search_training_epochs_previously_consumed"] == 20
    assert report["budget"]["search_training_epochs_this_command"] == 40
    assert report["budget"]["total_unique_training_epochs_this_command"] == 40
    with pytest.raises(SystemExit) as exc:
        child_launcher.main(arguments + ["--stop-after-search-epoch", "20"])
    assert exc.value.code == 2
