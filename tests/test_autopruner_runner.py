from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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


@pytest.mark.parametrize("full_recipe", [False, True])
def test_detached_child_receives_reused_baseline_checkpoint(tmp_path, monkeypatch, full_recipe):
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
    tuning_config = runner.FULL_TUNING_CONFIG if full_recipe else runner.DEFAULT_TUNING_CONFIG

    runner._start_detached(
        repeats_per_ratio=3,
        run_id="detached-reuse",
        log_path=log_path,
        status_path=status_path,
        baseline_checkpoint=checkpoint,
        tuning_config=tuning_config,
    )

    config_option = observed["command"].index("--tuning-config")
    assert observed["command"][config_option + 1] == tuning_config
    assert observed["command"][-2:] == [
        "--baseline-checkpoint",
        str(checkpoint),
    ]
    assert observed["kwargs"]["start_new_session"] is True
    active = json.loads(runner._active_run_path().read_text(encoding="utf-8"))
    assert active["pid"] == FakeProcess.pid
    runner._release_active_run(run_id="detached-reuse")


def test_runner_records_failed_tuning_command_in_status_and_log(tmp_path, monkeypatch):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    checkpoint = tmp_path / "existing" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    def fail_command(command, reporter):
        reporter.write("synthetic tuning traceback")
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(runner, "_run", fail_command)
    log_path = tmp_path / "outputs" / "logs" / "failed.log"
    status_path = log_path.with_suffix(".status.json")

    with pytest.raises(subprocess.CalledProcessError):
        runner._run_series(
            repeats_per_ratio=3,
            run_id="failed-tuning",
            log_path=log_path,
            status_path=status_path,
            mirror_to_console=False,
            baseline_checkpoint=checkpoint,
        )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["current_step"] == "autopruner_tuning"
    assert status["return_code"] == 17
    assert "src/net_complexity/tune.py" in status["failed_command"]
    log = log_path.read_text(encoding="utf-8")
    assert "synthetic tuning traceback" in log
    assert "AutoPruner series failed: CalledProcessError" in log


@pytest.mark.parametrize("full_recipe, expected_repeats", [(False, 3), (True, 1)])
def test_runner_recipe_defaults_preserve_repeat_count(monkeypatch, full_recipe, expected_repeats):
    runner = _load_runner_module()
    arguments = [str(RUNNER_PATH)]
    if full_recipe:
        arguments.extend(["--tuning-config", runner.FULL_TUNING_CONFIG])
    monkeypatch.setattr(sys, "argv", arguments)

    assert runner._parse_args().repeats_per_ratio == expected_repeats

    monkeypatch.setattr(sys, "argv", arguments + ["--repeats-per-ratio", "2"])
    assert runner._parse_args().repeats_per_ratio == 2


def test_full_series_trains_baseline_and_passes_checkpoint_to_six_run_grid(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir

    runner = _load_runner_module()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    commands = []
    checkpoint = tmp_path / "outputs/runs/full_best_practice_resnet50_on_cifar10/checkpoints/best.pt"

    def run_command(command, reporter):
        commands.append(command)
        if "src/net_complexity/train.py" in command:
            assert f"hydra.run.dir={checkpoint.parent.parent}" in command
            with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
                config = compose(config_name="train", overrides=command[3:])
            assert config.training_arguments.num_epochs == 200
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"baseline checkpoint")
        else:
            assert checkpoint.is_file()
            assert f"--config-name={runner.FULL_TUNING_CONFIG}" in command
            with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
                config = compose(config_name=runner.FULL_TUNING_CONFIG, overrides=command[4:])
            assert config.model.pretrained_checkpoint == str(checkpoint)
            assert config.training_arguments.num_epochs == 150
            assert config.model.final_fine_tune_epochs == 118
            assert config.tuning.repeats_per_trial == 1
            assert [p["model.backbone.target_keep_ratio"] for p in config.tuning.points] == [
                0.3, 0.5, 0.6, 0.7, 0.8, 0.9,
            ]

    monkeypatch.setattr(runner, "_run", run_command)
    log_path = tmp_path / "full.log"
    status_path = log_path.with_suffix(".status.json")
    runner._run_series(
        repeats_per_ratio=1,
        run_id="full",
        log_path=log_path,
        status_path=status_path,
        mirror_to_console=False,
        tuning_config=runner.FULL_TUNING_CONFIG,
    )

    assert len(commands) == 2
    status = json.loads(status_path.read_text())
    assert status["status"] == "completed"
    assert status["completed_steps"] == ["baseline_training", "autopruner_tuning"]
    assert status["tuning_config"] == runner.FULL_TUNING_CONFIG


@pytest.mark.parametrize("empty_checkpoint", [False, True])
def test_full_series_does_not_start_pruning_without_baseline_output(tmp_path, monkeypatch, empty_checkpoint):
    runner = _load_runner_module()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    commands = []

    def run_without_checkpoint(command, reporter):
        commands.append(command)
        if empty_checkpoint:
            checkpoint = tmp_path / "outputs/runs/missing_best_practice_resnet50_on_cifar10/checkpoints/best.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

    monkeypatch.setattr(runner, "_run", run_without_checkpoint)
    status_path = tmp_path / "missing.status.json"
    with pytest.raises(RuntimeError, match="without a non-empty best checkpoint"):
        runner._run_series(
            repeats_per_ratio=1,
            run_id="missing",
            log_path=tmp_path / "missing.log",
            status_path=status_path,
            mirror_to_console=False,
            tuning_config=runner.FULL_TUNING_CONFIG,
        )

    assert len(commands) == 1
    assert "src/net_complexity/train.py" in commands[0]
    assert json.loads(status_path.read_text())["status"] == "failed"
