from __future__ import annotations

import signal
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_pruning_pilot as launcher
from pruning_pilot_common import config_for
from net_complexity.training.pruning_audit import validate_config
from net_complexity.training.run_history import RunHistory


def _assert_production_channel_history_initializes(cfg, tmp_path):
    assert cfg.run_history.log_channel_history
    cfg.run_history.root_dir = str(tmp_path)
    cfg.run_history.use_hydra_output_dir = False
    history = RunHistory(cfg)
    assert history.channel_history_enabled
    assert history.channel_history_collector is not None
    assert history.channel_history_path.is_file()


def test_launcher_does_not_start_after_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: pytest.fail("process started"))
    with pytest.raises(TimeoutError):
        launcher.run_child(["unused"], tmp_path / "log", time.monotonic() - 1)


@pytest.mark.parametrize("needs_kill", [False, True])
def test_launcher_timeout_terms_trainer_then_kills_group_if_needed(tmp_path, monkeypatch, needs_kill):
    received = []
    class Process:
        pid = 12345
        count = 0
        def wait(self, timeout=None):
            self.count += 1
            if self.count == 1 or (needs_kill and self.count == 2):
                raise subprocess.TimeoutExpired("fake", timeout)
            return -15
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: received.append(("trainer", pid, sig)))
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: received.append(("group", pid, sig)))
    with pytest.raises(TimeoutError):
        launcher.run_child(["unused"], tmp_path / "log", time.monotonic() + 10)
    assert received[0] == ("trainer", 12345, signal.SIGTERM)
    assert len(received) == (2 if needs_kill else 1)
    if needs_kill:
        assert received[1] == ("group", 12345, signal.SIGKILL)


@pytest.mark.parametrize("job", launcher.JOBS + ["J4_internal_random"])
def test_all_pilot_configs_resolve_and_validate(monkeypatch, tmp_path, job):
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "/placeholder/not_loaded.pt")
    cfg = config_for(job)
    assert validate_config(cfg) == 150
    assert not cfg.training_arguments.evaluate_test
    assert not cfg.dataloaders.include_test
    assert not cfg.training_arguments.adaptive_lambda.enabled
    _assert_production_channel_history_initializes(cfg, tmp_path)


def test_unsupported_adaptive_and_typo_fail_closed(monkeypatch):
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "/placeholder/not_loaded.pt")
    cfg = config_for("J3_internal_fixed")
    cfg.training_arguments.adaptive_lambda.enabled = True
    with pytest.raises(ValueError, match="adaptive_lambda"):
        validate_config(cfg)
    cfg.training_arguments.adaptive_lambda.enabled = False
    from omegaconf import OmegaConf
    OmegaConf.update(cfg, "cyclic_channel_pruning.min_kepp_ratio", 0.2, force_add=True)
    with pytest.raises(ValueError, match="Unsupported"):
        validate_config(cfg)


@pytest.mark.parametrize("job", launcher.DAYTIME_JOBS + ["D3_internal_random"])
def test_daytime_is_one_cycle_and_one_seed(monkeypatch, tmp_path, job):
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "/placeholder/not_loaded.pt")
    cfg = config_for(job)
    assert validate_config(cfg) == 25
    assert cfg.seed == 42
    assert cfg.cyclic_channel_pruning.max_cycles == 1
    assert cfg.cyclic_channel_pruning.gumbel_epochs == 15
    assert cfg.cyclic_channel_pruning.final_epochs == 10
    assert not cfg.training_arguments.evaluate_test
    _assert_production_channel_history_initializes(cfg, tmp_path)
    if job != "D1_dense_control":
        assert cfg.cyclic_channel_pruning.max_param_fraction == 0.10
        assert cfg.cyclic_channel_pruning.min_keep_ratio == 0.50
        assert cfg.model.backbone.resnet_block.gate_internal_width
        assert not cfg.model.backbone.resnet_block.gate_output


def _completed_result(epochs=25, params=1000, accuracy=0.80):
    return {
        "status": "completed", "global_epochs_completed": epochs,
        "common_init_hash": "shared", "split_indices_hash": "shared_split",
        "optimizer_steps_total": epochs * 351,
        "validation": {"accuracy": accuracy, "ce_loss": 0.5, "example_count": 5000, "correct_count": int(accuracy * 5000)},
        "final_cost": {"physical_total_parameters": params, "conv_linear_macs_per_image": params * 100, "latency": None},
        "parameter_target_met": True, "decisions": [],
    }


def test_short_dense_does_not_inherit_nightly_92_percent_gate():
    result = _completed_result(accuracy=0.8)
    assert not launcher.dense_sanity_failed("daytime", result)
    assert launcher.dense_sanity_failed("nightly", result)


def test_comparison_keeps_actual_cost_and_validation_gap(tmp_path):
    for job, result in (
        ("D1_dense_control", _completed_result()),
        ("D2_internal_fixed", _completed_result(params=900, accuracy=0.78)),
    ):
        launcher.write_json(tmp_path / job / "pilot_state.json", result)
    report = launcher.write_comparison(tmp_path, launcher.DAYTIME_JOBS, "daytime")
    assert not report["test_evaluated"]
    assert report["runs"][1]["accuracy_delta_vs_dense_pp"] == pytest.approx(-2.0)
    assert report["runs"][1]["parameter_reduction_vs_dense"] == pytest.approx(0.1)
    assert (tmp_path / "comparison.md").is_file()


@pytest.mark.parametrize("profile", ["daytime", "nightly"])
@pytest.mark.parametrize("partial", [False, True])
def test_launcher_passes_profile_and_checks_its_epoch_budget(tmp_path, monkeypatch, profile, partial):
    from types import SimpleNamespace
    output = tmp_path / "run"
    calls = []
    monkeypatch.setattr(sys, "argv", ["launcher", "--output", str(output), "--profile", profile])
    # Do not falsify Python's version globally: OmegaConf also branches on it.
    monkeypatch.setattr(launcher, "sys", SimpleNamespace(
        version_info=(3, 10, 0), executable=sys.executable, version=sys.version))
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher, "provenance", lambda: {})
    monkeypatch.setattr(launcher.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(launcher, "make_initializer", lambda *a: None)
    expected_epochs = 25 if profile == "daytime" else 150

    def fake_child(command, log, deadline):
        if "--job" not in command:
            if any("smoke_pruning_pilot.py" in str(part) for part in command):
                assert command[command.index("--profile") + 1] == profile
            return
        assert command[command.index("--profile") + 1] == profile
        job = command[command.index("--job") + 1]
        calls.append(job)
        directory = Path(command[command.index("--output") + 1])
        result = _completed_result(epochs=expected_epochs - int(partial), accuracy=0.80 if profile == "daytime" else 0.94)
        launcher.write_json(directory / "pilot_state.json", result)
    monkeypatch.setattr(launcher, "run_child", fake_child)
    if partial:
        with pytest.raises(RuntimeError, match=f"exactly {expected_epochs}"):
            launcher.main()
    else:
        launcher.main()
        assert calls == launcher.PROFILE_JOBS[profile]
        assert (output / "comparison.json").is_file()
    status = json.loads((output / "nightly_status.json").read_text())
    assert status["profile"] == profile
    assert status["completed_jobs"] == ([] if partial else calls)
