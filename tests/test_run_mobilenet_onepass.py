from __future__ import annotations

import csv
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


def test_dry_run_prints_shared_reference_and_six_profiles_without_preflight(monkeypatch, capsys):
    monkeypatch.chdir(SCRIPT_PATH.parents[1])

    def fail_preflight(*args):
        pytest.fail("Dry-run must not import torch or check CUDA")

    monkeypatch.setattr(MODULE, "preflight", fail_preflight)
    assert MODULE.main(["--dry-run"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 7
    assert all(line.startswith(".venv/bin/python ") for line in lines)
    assert "training_arguments.num_epochs=30" in lines[0]
    assert "++training_arguments.evaluate_test=false" in lines[0]
    for cap, line in zip(MODULE.DEFAULT_CAPS, lines[1:]):
        assert f"threshold_max_param_fraction={cap}" in line
        assert "dense_reference_30ep" in line
        assert "--config-name mobilenet_onepass_thr05_gap05_1pp" in line


def test_existing_partial_stage_is_not_restarted(tmp_path):
    stage = MODULE.Stage("dense_reference", tmp_path / "dense", [], "expected")
    stage.directory.mkdir()
    MODULE.write_receipt(stage, "running")
    with pytest.raises(RuntimeError, match="incomplete or invalid.*refusing to repeat training"):
        MODULE.validate_existing_stage(stage)


def test_existing_foreign_or_changed_stage_is_not_reused(tmp_path):
    stage = MODULE.Stage("dense_reference", tmp_path / "dense", [], "expected")
    stage.directory.mkdir()
    with pytest.raises(RuntimeError, match="no launch receipt"):
        MODULE.validate_existing_stage(stage)
    MODULE.write_receipt(MODULE.Stage("dense_reference", stage.directory, [], "other"), "complete")
    with pytest.raises(RuntimeError, match="fingerprint differs"):
        MODULE.validate_existing_stage(stage)


def test_completed_reference_requires_all_thirty_validation_epochs(tmp_path):
    stage = MODULE.Stage("dense_reference", tmp_path / "dense", [], "expected")
    stage.directory.mkdir()
    MODULE.write_receipt(stage, "complete")
    (stage.directory / "summary.json").write_text(json.dumps({"timing": {"num_epochs_executed": 30}}))
    (stage.directory / "checkpoints").mkdir()
    (stage.directory / "checkpoints/best.pt").write_bytes(b"checkpoint")
    with (stage.directory / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "valid_accuracy"])
        writer.writeheader()
        writer.writerows({"epoch": epoch, "valid_accuracy": 0.7} for epoch in range(1, 31))
    assert MODULE.validate_existing_stage(stage)
    with (stage.directory / "history.csv").open("a") as handle:
        handle.write("31,0.7\n")
    with pytest.raises(RuntimeError, match="refusing to repeat training"):
        MODULE.validate_existing_stage(stage)


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
