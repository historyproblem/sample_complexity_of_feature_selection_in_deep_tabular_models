from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_pruning_pilot as launcher
from pruning_pilot_common import config_for
from net_complexity.training.pruning_audit import validate_config


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
def test_all_pilot_configs_resolve_and_validate(monkeypatch, job):
    monkeypatch.setenv("AUDIT_INIT_CHECKPOINT", "/placeholder/not_loaded.pt")
    cfg = config_for(job)
    assert validate_config(cfg) == 150
    assert not cfg.training_arguments.evaluate_test
    assert not cfg.dataloaders.include_test
    assert not cfg.training_arguments.adaptive_lambda.enabled


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
