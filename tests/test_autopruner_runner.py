from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/run_autopruner_cifar10_v100_11h.py"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("autopruner_v100_runner", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_captures_child_stdout_and_stderr(tmp_path):
    runner = _load_runner_module()
    log_path = tmp_path / "autopruner.log"
    reporter = runner.RunReporter(log_path, mirror_to_console=False)

    try:
        runner._run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "print('autopruner-stdout', flush=True); "
                    "print('autopruner-stderr', file=sys.stderr, flush=True)"
                ),
            ],
            reporter,
        )
    finally:
        reporter.close()

    log = log_path.read_text(encoding="utf-8")
    assert "autopruner-stdout" in log
    assert "autopruner-stderr" in log


def test_runner_writes_machine_readable_status_atomically(tmp_path):
    runner = _load_runner_module()
    status_path = tmp_path / "autopruner.status.json"
    payload = {"status": "running", "pid": 123, "current_step": "baseline_training"}

    runner._write_status(status_path, payload)

    assert json.loads(status_path.read_text(encoding="utf-8")) == payload
    assert not status_path.with_suffix(status_path.suffix + ".tmp").exists()


def test_runner_prevents_a_second_live_series(tmp_path, monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)

    active_path = runner._claim_active_run(run_id="first", pid=os.getpid())
    with pytest.raises(RuntimeError, match="already running"):
        runner._claim_active_run(run_id="second", pid=os.getpid())

    runner._release_active_run(run_id="first")
    assert not active_path.exists()


def test_runner_reuses_existing_baseline_without_retraining(tmp_path, monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    checkpoint = tmp_path / "existing_run" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    commands = []

    def record_command(command, reporter):
        del reporter
        commands.append(command)

    monkeypatch.setattr(runner, "_run", record_command)
    log_path = tmp_path / "outputs" / "logs" / "reuse.log"
    status_path = log_path.with_suffix(".status.json")

    runner._run_series(
        repeats_per_ratio=3,
        run_id="reuse",
        log_path=log_path,
        status_path=status_path,
        mirror_to_console=False,
        baseline_checkpoint=checkpoint,
    )

    assert len(commands) == 1
    assert "src/net_complexity/tune.py" in commands[0]
    assert f"model.pretrained_checkpoint={checkpoint}" in commands[0]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["baseline_reused"] is True
    assert status["completed_steps"] == [
        "baseline_checkpoint_reused",
        "autopruner_tuning",
    ]


def test_detached_child_receives_reused_baseline_checkpoint(tmp_path, monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    checkpoint = tmp_path / "existing" / "checkpoints" / "best.pt"
    observed = {}

    class FakeProcess:
        pid = 43210

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    log_path = tmp_path / "outputs" / "logs" / "detached.log"
    status_path = log_path.with_suffix(".status.json")

    runner._start_detached(
        repeats_per_ratio=3,
        run_id="detached-reuse",
        log_path=log_path,
        status_path=status_path,
        baseline_checkpoint=checkpoint,
    )

    assert observed["command"][-2:] == [
        "--baseline-checkpoint",
        str(checkpoint),
    ]
    assert observed["kwargs"]["start_new_session"] is True
    active = json.loads(runner._active_run_path().read_text(encoding="utf-8"))
    assert active["pid"] == FakeProcess.pid
    runner._release_active_run(run_id="detached-reuse")
