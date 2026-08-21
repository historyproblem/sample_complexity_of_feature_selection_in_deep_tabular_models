from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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
