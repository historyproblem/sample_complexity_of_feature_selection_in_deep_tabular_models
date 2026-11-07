from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/run_mobilenet_onepass.py"
SPEC = importlib.util.spec_from_file_location("run_mobilenet_onepass", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_dry_run_prints_archive_import_and_six_profiles_without_preflight(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(SCRIPT_PATH.parents[1])

    def fail_preflight(*args):
        pytest.fail("Dry-run must not import torch or check CUDA")

    monkeypatch.setattr(MODULE, "preflight", fail_preflight)
    missing_archive = tmp_path / "not_uploaded.tar.gz"
    assert MODULE.main(["--dry-run", "--baseline-archive", str(missing_archive)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7
    assert all(line.startswith(".venv/bin/python ") for line in lines)
    assert "scripts/prepare_mobilenet_reference.py --archive" in lines[0]
    assert str(missing_archive) in lines[0]
    assert not any("src/net_complexity/cli/train.py" in line for line in lines)
    for cap, line in zip(MODULE.DEFAULT_CAPS, lines[1:]):
        assert f"threshold_max_param_fraction={cap}" in line
        assert "imported_dense_reference_20261002" in line
        assert "--config-name mobilenet_onepass_thr05_gap05_1pp" in line


def test_existing_partial_stage_is_not_restarted(tmp_path):
    stage = MODULE.Stage("onepass", tmp_path / "profile", [], "expected")
    stage.directory.mkdir()
    MODULE.write_receipt(stage, "running")
    with pytest.raises(RuntimeError, match="incomplete or invalid.*refusing to repeat training"):
        MODULE.validate_existing_stage(stage)


def test_existing_foreign_or_changed_stage_is_not_reused(tmp_path):
    stage = MODULE.Stage("onepass", tmp_path / "profile", [], "expected")
    stage.directory.mkdir()
    with pytest.raises(RuntimeError, match="no launch receipt"):
        MODULE.validate_existing_stage(stage)
    MODULE.write_receipt(MODULE.Stage("onepass", stage.directory, [], "other"), "complete")
    with pytest.raises(RuntimeError, match="fingerprint differs"):
        MODULE.validate_existing_stage(stage)


def test_imported_reference_keeps_full_history_and_actual_cosine150_recipe(tmp_path, monkeypatch):
    archive_path = tmp_path / "selected.tar.gz"
    archive_path.write_bytes(b"selected historical archive")
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    monkeypatch.setattr(MODULE, "EXPECTED_ARCHIVE_SHA256", archive_hash)
    reference = tmp_path / "reference"
    reference.mkdir()
    history_path = reference / "history.csv"
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "valid_accuracy"])
        writer.writeheader()
        writer.writerows({"epoch": epoch, "valid_accuracy": 0.7} for epoch in range(1, 87))
    metadata = {
        "archive_sha256": archive_hash, "actual_training_horizon_epochs": 150,
        "actual_cosine_T_max": 150, "available_history_epochs": 86,
        "selected_reference_epochs": [1, 30], "training_performed": False,
        "history_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
    }
    provenance_path = reference / "reference_provenance.json"
    provenance_path.write_text(json.dumps(metadata))
    identity = MODULE.validate_imported_reference(reference, archive_path)
    assert identity["available_history_epochs"] == 86
    assert identity["actual_cosine_T_max"] == 150
    metadata["actual_training_horizon_epochs"] = 30
    provenance_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="unexpected actual_training_horizon_epochs"):
        MODULE.validate_imported_reference(reference, archive_path)


def test_missing_archive_stops_before_cuda_or_training(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(SCRIPT_PATH.parents[1])

    def fail_preflight(*args):
        pytest.fail("Missing archive must stop before CUDA or any training")

    monkeypatch.setattr(MODULE, "preflight", fail_preflight)
    assert MODULE.main(["--baseline-archive", str(tmp_path / "not_uploaded.tar.gz")]) == 1
    assert "No new dense baseline will be trained" in capsys.readouterr().err


def test_profile_fingerprints_include_reference_identity():
    root = SCRIPT_PATH.parents[1]
    first = MODULE.build_plan(root, ["0.1"], {"archive_sha256": "archiveA", "history_sha256": "historyA"})
    other = MODULE.build_plan(root, ["0.1"], {"archive_sha256": "archiveA", "history_sha256": "historyB"})
    assert len(first) == 1
    assert first[0].fingerprint != other[0].fingerprint


def test_completed_profile_must_have_exact_budget_and_frozen_test(tmp_path):
    stage = MODULE.Stage("onepass", tmp_path / "profile", [], "expected")
    final = stage.directory / "final/final_run"
    final.mkdir(parents=True)
    MODULE.write_receipt(stage, "complete")
    (final / "checkpoints").mkdir()
    (final / "checkpoints/best.pt").write_bytes(b"checkpoint")
    with (final / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "valid_accuracy"])
        writer.writeheader()
        writer.writerows({"epoch": epoch, "valid_accuracy": 0.7} for epoch in range(1, 46))
    summary = {
        "num_cycles_completed": 1,
        "training_cost": {"num_epochs_executed": 75},
        "cycle_results": [{"num_epochs_executed": 30}],
        "recovery_results": [{"num_epochs_executed": 45}],
        "final_result": {"run_dir": str(final), "test_metrics": {"test_accuracy": 0.7}},
    }
    summary_path = stage.directory / "cyclic_summary.json"
    summary_path.write_text(json.dumps(summary))
    assert MODULE.validate_existing_stage(stage)
    summary["training_cost"]["num_epochs_executed"] = 76
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="refusing to repeat training"):
        MODULE.validate_existing_stage(stage)
    summary["training_cost"]["num_epochs_executed"] = 75
    summary["final_result"]["test_metrics"] = {}
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="no frozen-checkpoint test"):
        MODULE.validate_existing_stage(stage)


def test_launcher_lock_excludes_second_process_and_releases_after_exit(tmp_path):
    with MODULE.exclusive_launcher(tmp_path):
        with pytest.raises(RuntimeError, match="Another MobileNet launcher"):
            with MODULE.exclusive_launcher(tmp_path):
                pytest.fail("Second launcher acquired the same output lock")
    with MODULE.exclusive_launcher(tmp_path):
        pass
