"""Progress is flushed during blocked work and is independent of model execution."""
from __future__ import annotations

from copy import deepcopy
import builtins
import importlib.util
from pathlib import Path
import runpy
import subprocess
import sys
import threading

import pytest


PROGRESS_PATH = (Path(__file__).resolve().parents[1]
                 / "src/net_complexity/training/one_shot_progress.py")


def _progress_module():
    # Load the logging helper in isolation; package __init__ eagerly imports ML.
    spec = importlib.util.spec_from_file_location("isolated_one_shot_progress", PROGRESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def printed(monkeypatch):
    records = []
    heartbeat = threading.Event()
    after_exit = threading.Event()
    exited = threading.Event()
    owner = threading.get_ident()

    def capture(*args, **kwargs):
        records.append((" ".join(str(arg) for arg in args), kwargs))
        if threading.get_ident() != owner:
            heartbeat.set()
            if exited.is_set():
                after_exit.set()

    monkeypatch.setattr("builtins.print", capture)
    return records, heartbeat, after_exit, exited


def test_phase_start_is_flushed_before_work_and_heartbeat_during_block(printed):
    progress = _progress_module()
    records, heartbeat, after_exit, exited = printed
    with progress.phase_progress("data", "prepare CIFAR train/validation", interval_seconds=0.01):
        assert records, "The user must see progress before data preparation begins"
        assert "data" in records[0][0] and "prepare CIFAR" in records[0][0]
        assert records[0][1].get("flush") is True
        assert heartbeat.wait(timeout=1), "Blocked phase must still emit a liveness heartbeat"
    exited.set()
    assert not after_exit.wait(timeout=0.04), "Context exit must stop and join its heartbeat"
    assert all(kwargs.get("flush") is True for _, kwargs in records)
    assert "data" in records[-1][0]


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_phase_exception_is_preserved_and_heartbeat_stops(printed, error_type):
    progress = _progress_module()
    records, heartbeat, after_exit, exited = printed
    failure = error_type("deliberate data failure")
    with pytest.raises(error_type, match="deliberate data failure") as caught:
        with progress.phase_progress("export_only", "measure logits", interval_seconds=0.01):
            assert heartbeat.wait(timeout=1)
            raise failure
    exited.set()
    assert caught.value is failure
    assert not after_exit.wait(timeout=0.04)
    assert "export_only" in records[-1][0]
    assert "failed" in records[-1][0].lower()
    assert all(kwargs.get("flush") is True for _, kwargs in records)


def test_live_ledger_progress_and_epoch_metrics_are_observational(printed):
    progress = _progress_module()
    records, heartbeat, _, _ = printed
    ledger = {"global_training_epoch": 60, "search_epochs_consumed": 60,
              "optimizer_updates": 120, "consumed_training_examples": 480}
    with progress.phase_progress("inherited", "train compact model", interval_seconds=0.01,
                                 ledger=ledger, total_epochs=90):
        ledger.update(global_training_epoch=61, optimizer_updates=122,
                      consumed_training_examples=488)
        expected = deepcopy(ledger)
        assert heartbeat.wait(timeout=1)
        valid = {"valid_accuracy": 0.8, "valid_ce_loss": 0.45,
                 "valid_correct_count": 4, "valid_example_count": 5}
        valid_before = deepcopy(valid)
        progress.epoch_progress("inherited", 1, 90, valid, ledger, elapsed_seconds=2.5)
        assert ledger == expected and valid == valid_before
    epoch_lines = [line for line, _ in records if "valid_accuracy=80.0000%" in line]
    assert epoch_lines, "Epoch indication must include actual validation accuracy"
    assert any("0.45" in line and "122" in line and "488" in line for line in epoch_lines)
    assert any("1/90" in line for line, _ in records), "Physical progress must use stage epoch offset"
    assert all(kwargs.get("flush") is True for _, kwargs in records)


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_interval_cannot_start_busy_heartbeat(interval):
    with pytest.raises(ValueError, match="finite and positive"):
        with _progress_module().phase_progress("data", "prepare", interval_seconds=interval):
            pytest.fail("Invalid heartbeat interval must fail before phase body")


def test_closed_output_does_not_abort_work_or_mask_its_error(monkeypatch):
    progress = _progress_module()

    def broken_output(*args, **kwargs):
        raise BrokenPipeError("terminal closed")

    monkeypatch.setattr("builtins.print", broken_output)
    completed = []
    with progress.phase_progress("data", "prepare", interval_seconds=0.005):
        completed.append(True)
        threading.Event().wait(0.02)
    assert completed == [True]
    with pytest.raises(ValueError, match="actual phase failure"):
        with progress.phase_progress("data", "prepare", interval_seconds=0.005):
            raise ValueError("actual phase failure")


def test_progress_import_and_heartbeat_do_not_load_ml_or_consume_rng():
    script = r'''
import importlib.util
import random
import sys
import threading

random.seed(12345)
before = random.getstate()
spec = importlib.util.spec_from_file_location("progress_standalone", sys.argv[1])
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)
with progress.phase_progress("imports", "loading runtime", interval_seconds=0.005):
    threading.Event().wait(0.02)
assert random.getstate() == before
assert not any(name == "torch" or name.startswith("torch.")
               or name == "numpy" or name.startswith("numpy.") for name in sys.modules)
print("standalone-no-ml-no-rng-change")
'''
    result = subprocess.run([sys.executable, "-c", script, str(PROGRESS_PATH)],
                            check=True, capture_output=True, text=True, timeout=10)
    assert "standalone-no-ml-no-rng-change" in result.stdout


def test_launcher_flushes_startup_before_importing_model_package(monkeypatch, printed):
    records, _, _, _ = printed
    launcher = Path(__file__).resolve().parents[1] / "scripts/launch_one_shot_pruning.py"
    monkeypatch.setattr(sys, "argv", [str(launcher), "--from-scratch"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    original_import = builtins.__import__

    def stop_at_model_package(name, *args, **kwargs):
        if name.startswith("net_complexity"):
            assert records, "Startup must be visible even while the first package import blocks"
            assert "Starting launcher" in records[0][0]
            assert records[0][1].get("flush") is True
            raise ImportError("deliberate stop before model package initialization")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stop_at_model_package)
    with pytest.raises(ImportError, match="deliberate stop before model"):
        runpy.run_path(str(launcher), run_name="__main__")
