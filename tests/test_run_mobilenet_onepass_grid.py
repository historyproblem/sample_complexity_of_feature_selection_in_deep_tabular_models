from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import shutil
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/run_mobilenet_onepass_grid.py"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_mobilenet_onepass_grid", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _overrides(command):
    return {
        key: value
        for argument in command
        if "=" in argument
        for key, value in [argument.split("=", 1)]
    }


def test_grid_plan_has_exact_cross_product_and_eight_distinct_outputs():
    root = SCRIPT_PATH.parents[1]
    plan = MODULE.build_plan(root)
    observed = set()
    for stage in plan:
        overrides = _overrides(stage.command)
        observed.add((
            float(overrides["cyclic_channel_pruning.g_prob_threshold"]),
            float(overrides["training_arguments.adaptive_lambda.soft_drop"]),
            float(overrides["training_arguments.adaptive_lambda.hard_drop"]),
        ))
        assert overrides["cyclic_channel_pruning.threshold_max_param_fraction"] == "null"
        assert overrides["++cyclic_channel_pruning.search_only"] == "true"
        assert overrides["cyclic_channel_pruning.final_epochs"] == "0"
        assert overrides["cyclic_channel_pruning.recovery_epochs"] == "0"
        assert overrides["training_arguments.adaptive_lambda.warmup_epochs"] == "10"
        assert float(overrides[
            "++training_arguments.adaptive_lambda.auto_log_step_epoch_fraction"
        ]) == 1.0
        assert stage.kind == "search_only"
        assert stage.directory.name.endswith("_search_only_warmup10")
        assert overrides.get("++seed", overrides.get("+seed")) == "42"
        assert not any(
            argument.startswith("model.backbone.block.gate_threshold=")
            for argument in stage.command
        )
        assert stage.command[:3] == [
            ".venv/bin/python", "-u", "src/net_complexity/cli/cyclic_train.py",
        ]
    expected = {
        (threshold, soft, hard)
        for threshold, (soft, hard) in itertools.product(
            (0.5, 0.9), ((0.04, 0.08), (0.06, 0.10), (0.08, 0.12), (0.10, 0.14)),
        )
    }
    assert len(plan) == len(observed) == 8
    assert observed == expected
    assert len({stage.directory for stage in plan}) == 8
    assert len({stage.fingerprint for stage in plan}) == 8


def test_grid_dry_run_needs_neither_archive_nor_cuda(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(SCRIPT_PATH.parents[1])

    def unexpected_preflight(*args, **kwargs):
        pytest.fail("Dry-run must not check CUDA or start training")

    monkeypatch.setattr(MODULE.single, "preflight", unexpected_preflight)
    missing_archive = tmp_path / "not_uploaded.tar.gz"
    assert MODULE.main(["--dry-run", "--baseline-archive", str(missing_archive)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 9
    assert all(line.startswith(".venv/bin/python ") for line in lines)
    assert "scripts/prepare_mobilenet_reference.py" in lines[0]
    assert str(missing_archive) in lines[0]
    assert all("src/net_complexity/cli/cyclic_train.py" in line for line in lines[1:])
    assert not any("src/net_complexity/cli/train.py" in line for line in lines)


def test_grid_fingerprints_include_reference_identity():
    root = SCRIPT_PATH.parents[1]
    first = MODULE.build_plan(root, {"archive_sha256": "a", "history_sha256": "h1"})
    changed = MODULE.build_plan(root, {"archive_sha256": "a", "history_sha256": "h2"})
    assert len(first) == len(changed) == 8
    assert all(a.directory == b.directory for a, b in zip(first, changed))
    assert all(a.fingerprint != b.fingerprint for a, b in zip(first, changed))


@pytest.mark.parametrize("invalid_case", ["duplicate_band", "reversed_band", "missing_required_band"])
def test_grid_rejects_invalid_or_incomplete_lambda_comparisons(tmp_path, invalid_case):
    grid = MODULE.load_grid(SCRIPT_PATH.parents[1])
    if invalid_case == "duplicate_band":
        grid["lambda_bands"][0] = dict(grid["lambda_bands"][1])
    elif invalid_case == "reversed_band":
        grid["lambda_bands"][0] = {"soft_drop": 0.08, "hard_drop": 0.04}
    else:
        grid["lambda_bands"][2] = {"soft_drop": 0.09, "hard_drop": 0.13}
    target = tmp_path / MODULE.GRID_CONFIG_PATH
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(grid), encoding="utf-8")
    with pytest.raises(ValueError):
        MODULE.load_grid(tmp_path)


@pytest.fixture
def launch_context(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / "scripts").mkdir()
    original_root = SCRIPT_PATH.parents[1]
    for relative in (
        Path("scripts/run_mobilenet_onepass_grid.py"),
        Path("scripts/run_mobilenet_onepass.py"),
        Path("configs/mobilenet_onepass_thr05_gap8_12pp.yaml"),
        MODULE.GRID_CONFIG_PATH,
    ):
        shutil.copyfile(original_root / relative, root / relative)
    monkeypatch.setattr(MODULE, "__file__", str(root / "scripts/run_mobilenet_onepass_grid.py"))
    monkeypatch.setattr(MODULE.subprocess, "check_output", lambda *args, **kwargs: "test-revision\n")
    monkeypatch.chdir(root)
    archive = root / "baseline.tar.gz"
    archive.write_bytes(b"accepted baseline")
    identity = {"archive_sha256": "accepted", "history_sha256": "history"}
    reference_checks = []

    def validate_reference(*args, **kwargs):
        reference_checks.append(True)
        return dict(identity)

    monkeypatch.setattr(MODULE.single, "validate_imported_reference", validate_reference)
    monkeypatch.setattr(MODULE.single, "preflight", lambda *args, **kwargs: None)
    plan = MODULE.build_plan(root, identity)
    return root, archive, identity, plan, reference_checks


def _write_completed_training(directory):
    search = directory / "cycle_0/synthetic_run"
    (search / "checkpoints").mkdir(parents=True)
    (search / "checkpoints/best.pt").write_bytes(b"best checkpoint")
    (search / "checkpoints/last.pt").write_bytes(b"last checkpoint")
    with (search / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "valid_accuracy"])
        writer.writeheader()
        writer.writerows({"epoch": epoch, "valid_accuracy": 0.7} for epoch in range(1, 31))
    mask = {"backbone.features.3.mid_gumbel_layer": [0]}
    mask_path = directory / "channel_mask.json"
    architecture_path = directory / "architecture.json"
    mask_path.write_text(json.dumps(mask), encoding="utf-8")
    architecture = {
        "total_params": 2480011,
        "dense_total_params": 2480072,
        "removed_params": 61,
        "removed_fraction": 61 / 2480072,
        "channels_removed": 1,
        "mask_file": str(mask_path),
        "architecture_file": str(architecture_path),
        "structural": True,
        "trained": False,
        "gate_free": True,
    }
    architecture_path.write_text(json.dumps({
        **architecture,
        "purpose": "architecture_counts_only",
        "weights_exported": False,
        "channel_mask": mask,
        "geometry": [{"name": "backbone.features.3.branch.depthwise", "in_channels": 143}],
    }), encoding="utf-8")
    summary = {
        "search_only": True,
        "num_cycles_completed": 1,
        "training_cost": {"num_epochs_executed": 30},
        "cycle_results": [{
            "num_epochs_executed": 30,
            "run_dir": str(search),
            "last_valid_metrics": {"valid_accuracy": 0.7},
        }],
        "recovery_results": [],
        "final_result": None,
        "final_disabled_channels": mask,
        "architecture": architecture,
    }
    (directory / "cyclic_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_grid_executes_eight_search_only_profiles_sequentially(monkeypatch, launch_context):
    root, archive, identity, plan, reference_checks = launch_context
    training_commands = []
    all_commands = []

    def run_child(command, **kwargs):
        all_commands.append(command)
        assert kwargs["cwd"] == root
        assert kwargs["check"] is True
        if "src/net_complexity/cli/cyclic_train.py" in command:
            directory = Path(json.loads(_overrides(command)["hydra.run.dir"]))
            assert directory == plan[len(training_commands)].directory
            for previous in plan[:len(training_commands)]:
                assert MODULE.validate_existing_stage(previous)
                receipt = json.loads((previous.directory / "launcher_receipt.json").read_text())
                assert receipt["status"] == "complete"
            training_commands.append(command)
            _write_completed_training(directory)
        return MODULE.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.subprocess, "run", run_child)
    assert MODULE.main(["--baseline-archive", str(archive)]) == 0
    assert len(all_commands) == 9
    assert training_commands == [stage.command for stage in plan]
    assert len(reference_checks) >= 9
    for stage in plan:
        assert MODULE.validate_existing_stage(stage)
    summary = json.loads((root / MODULE.load_grid(root)["output_root"] / "grid_summary.json").read_text())
    assert summary["total_new_epochs"] == 240
    assert len(summary["profiles"]) == 8
    assert all("test_metrics" not in profile for profile in summary["profiles"])
    assert all(profile["num_epochs_executed"] == 30 for profile in summary["profiles"])
    assert all(profile["last_search_validation_accuracy"] == 0.7 for profile in summary["profiles"])
    assert all(profile["architecture"]["total_params"] == 2480011 for profile in summary["profiles"])
    assert all(profile["architecture"]["removed_params"] == 61 for profile in summary["profiles"])

    # The second invocation may re-check/import metadata, but must not train
    # any of the eight completed, matching profiles again.
    assert MODULE.main(["--baseline-archive", str(archive)]) == 0
    assert len(all_commands) == 10
    assert len(training_commands) == 8


@pytest.mark.parametrize("corruption", [
    "missing_mask", "missing_architecture", "missing_last_checkpoint",
    "inconsistent_parameter_counts", "mismatched_mask", "recovery_present", "old_epoch_budget",
])
def test_search_completion_requires_real_search_and_matching_architecture_artifacts(
    launch_context, corruption,
):
    root, archive, identity, plan, reference_checks = launch_context
    stage = plan[0]
    stage.directory.mkdir(parents=True)
    MODULE.single.write_receipt(stage, "complete")
    _write_completed_training(stage.directory)
    assert MODULE.validate_existing_stage(stage)
    summary_path = stage.directory / "cyclic_summary.json"
    summary = json.loads(summary_path.read_text())
    architecture_path = Path(summary["architecture"]["architecture_file"])
    mask_path = Path(summary["architecture"]["mask_file"])
    if corruption == "missing_mask":
        mask_path.unlink()
    elif corruption == "missing_architecture":
        architecture_path.unlink()
    elif corruption == "missing_last_checkpoint":
        (Path(summary["cycle_results"][0]["run_dir"]) / "checkpoints/last.pt").unlink()
    elif corruption == "inconsistent_parameter_counts":
        summary["architecture"]["removed_params"] += 1
        stored = json.loads(architecture_path.read_text())
        stored["removed_params"] = summary["architecture"]["removed_params"]
        architecture_path.write_text(json.dumps(stored), encoding="utf-8")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    elif corruption == "mismatched_mask":
        mask_path.write_text(json.dumps({"backbone.features.3.mid_gumbel_layer": [0, 1]}))
    elif corruption == "recovery_present":
        summary["recovery_results"] = [{"num_epochs_executed": 45}]
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    else:
        summary["training_cost"]["num_epochs_executed"] = 75
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError):
        MODULE.validate_existing_stage(stage)


def test_last_partial_profile_blocks_training_of_every_profile(monkeypatch, launch_context, capsys):
    root, archive, identity, plan, reference_checks = launch_context
    partial = plan[-1]
    partial.directory.mkdir(parents=True)
    MODULE.single.write_receipt(partial, "running")
    training_commands = []

    def run_child(command, **kwargs):
        if "src/net_complexity/cli/cyclic_train.py" in command:
            training_commands.append(command)
        return MODULE.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.subprocess, "run", run_child)
    assert MODULE.main(["--baseline-archive", str(archive)]) == 1
    assert training_commands == []
    assert "refusing" in capsys.readouterr().err
    assert not plan[0].directory.exists()


def test_grid_stops_at_first_failed_profile(monkeypatch, launch_context):
    root, archive, identity, plan, reference_checks = launch_context
    training_commands = []

    def run_child(command, **kwargs):
        if "src/net_complexity/cli/cyclic_train.py" in command:
            training_commands.append(command)
            if len(training_commands) == 3:
                raise MODULE.subprocess.CalledProcessError(1, command)
            directory = Path(json.loads(_overrides(command)["hydra.run.dir"]))
            _write_completed_training(directory)
        return MODULE.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.subprocess, "run", run_child)
    assert MODULE.main(["--baseline-archive", str(archive)]) == 1
    assert training_commands == [stage.command for stage in plan[:3]]
    assert all(MODULE.validate_existing_stage(stage) for stage in plan[:2])
    assert (plan[2].directory / "launcher_receipt.json").is_file()
    assert not any(stage.directory.exists() for stage in plan[3:])


def test_grid_checks_reference_again_before_each_training_child(monkeypatch, launch_context):
    root, archive, identity, plan, reference_checks = launch_context
    training_commands = []
    checks = []

    def validate_reference(*args, **kwargs):
        checks.append(True)
        if len(checks) >= 3:
            return {**identity, "history_sha256": "changed"}
        return dict(identity)

    def run_child(command, **kwargs):
        if "src/net_complexity/cli/cyclic_train.py" in command:
            training_commands.append(command)
            directory = Path(json.loads(_overrides(command)["hydra.run.dir"]))
            _write_completed_training(directory)
        return MODULE.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.single, "validate_imported_reference", validate_reference)
    monkeypatch.setattr(MODULE.subprocess, "run", run_child)
    assert MODULE.main(["--baseline-archive", str(archive)]) == 1
    assert training_commands == [plan[0].command]
    assert not plan[1].directory.exists()
