from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]


class RunReporter:
    def __init__(self, log_path: Path, *, mirror_to_console: bool = True) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self.mirror_to_console = mirror_to_console
        self._handle: TextIO = log_path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        self._handle.close()

    def write(self, message: str, *, end: str = "\n") -> None:
        self._handle.write(message + end)
        self._handle.flush()
        if self.mirror_to_console:
            print(message, end=end, flush=True)


def _run(command: list[str], reporter: RunReporter) -> None:
    reporter.write("+ " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        reporter.write(line, end="")
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _write_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = status_path.with_suffix(status_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(status_path)


def _active_run_path() -> Path:
    return REPO_ROOT / "outputs" / "logs" / "autopruner_v100_11h.active.json"


def _pid_is_alive(pid: Any) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _claim_active_run(*, run_id: str, pid: int) -> Path:
    """Atomically prevent two expensive V100 series from sharing one GPU."""

    active_path = _active_run_path()
    active_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "pid": int(pid),
        "claimed_at": _timestamp(),
    }
    while True:
        try:
            descriptor = os.open(
                active_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            try:
                existing = json.loads(active_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if existing.get("run_id") == run_id:
                _write_status(active_path, payload)
                return active_path
            if _pid_is_alive(existing.get("pid")):
                raise RuntimeError(
                    "Another AutoPruner V100 series is already running: "
                    f"run_id={existing.get('run_id')} pid={existing.get('pid')}. "
                    f"Active-run file: {active_path}"
                )
            try:
                active_path.unlink()
            except FileNotFoundError:
                pass
            continue

        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return active_path


def _release_active_run(*, run_id: str) -> None:
    active_path = _active_run_path()
    try:
        existing = json.loads(active_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    if existing.get("run_id") != run_id:
        return
    try:
        active_path.unlink()
    except FileNotFoundError:
        pass


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the controlled CIFAR-10 ResNet-50 baseline and run the "
            "AutoPruner keep-ratio series on its best checkpoint."
        )
    )
    parser.add_argument(
        "--repeats-per-ratio",
        type=int,
        default=3,
        help=(
            "Repeats for each author-reported keep ratio. The default of 3 "
            "leaves room for baseline training in one 11-hour V100 job."
        ),
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help=(
            "Start the series in a new session and return immediately. The "
            "runner prints the PID, durable log path, and status path."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Durable combined stdout/stderr log. By default a timestamped file "
            "is created under outputs/logs."
        ),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=None,
        help=(
            "Reuse an existing non-empty baseline best.pt and skip the 200-epoch "
            "baseline training step."
        ),
    )
    parser.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-console-mirror",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _start_detached(
    *,
    repeats_per_ratio: int,
    run_id: str,
    log_path: Path,
    status_path: Path,
    baseline_checkpoint: Path | None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--repeats-per-ratio",
        str(repeats_per_ratio),
        "--run-id",
        run_id,
        "--log-file",
        str(log_path),
        "--no-console-mirror",
    ]
    if baseline_checkpoint is not None:
        command.extend(["--baseline-checkpoint", str(baseline_checkpoint)])
    active_path = _claim_active_run(run_id=run_id, pid=os.getpid())
    _write_status(
        status_path,
        {
            "status": "launching",
            "run_id": run_id,
            "pid": None,
            "started_at": _timestamp(),
            "log_file": str(log_path),
            "status_file": str(status_path),
        },
    )
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(f"[{_timestamp()}] launching detached: {' '.join(command)}\n")
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _write_status(
                active_path,
                {
                    "run_id": run_id,
                    "pid": process.pid,
                    "claimed_at": _timestamp(),
                },
            )
    except BaseException as error:
        _release_active_run(run_id=run_id)
        _write_status(
            status_path,
            {
                "status": "failed_to_launch",
                "run_id": run_id,
                "pid": None,
                "finished_at": _timestamp(),
                "log_file": str(log_path),
                "status_file": str(status_path),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise

    print(f"AutoPruner detached process started: pid={process.pid}", flush=True)
    print(f"Log: {log_path}", flush=True)
    print(f"Status: {status_path}", flush=True)


def _run_series(
    *,
    repeats_per_ratio: int,
    run_id: str,
    log_path: Path,
    status_path: Path,
    mirror_to_console: bool,
    baseline_checkpoint: Path | None = None,
) -> None:
    started_at = _timestamp()

    if baseline_checkpoint is None:
        baseline_dir = (
            REPO_ROOT
            / "outputs"
            / "runs"
            / f"{run_id}_best_practice_resnet50_on_cifar10"
        ).resolve()
        checkpoint = baseline_dir / "checkpoints" / "best.pt"
    else:
        checkpoint = baseline_checkpoint.resolve()
        baseline_dir = checkpoint.parent.parent
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(
                "--baseline-checkpoint must point to a non-empty file: "
                f"{checkpoint}"
            )
    reporter = RunReporter(log_path, mirror_to_console=mirror_to_console)
    baseline_reused = baseline_checkpoint is not None
    status: dict[str, Any] = {
        "status": "running",
        "run_id": run_id,
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": started_at,
        "current_step": "autopruner_tuning" if baseline_reused else "baseline_training",
        "completed_steps": ["baseline_checkpoint_reused"] if baseline_reused else [],
        "baseline_dir": str(baseline_dir),
        "checkpoint": str(checkpoint),
        "baseline_reused": baseline_reused,
        "log_file": str(log_path),
        "status_file": str(status_path),
        "mlflow_tracking_uri": f"sqlite:///{(REPO_ROOT / 'mlflow.db').resolve()}",
    }
    _write_status(status_path, status)
    reporter.write(f"[{started_at}] AutoPruner series started (pid={os.getpid()})")
    reporter.write(f"Durable log: {log_path}")
    reporter.write(f"Status file: {status_path}")

    try:
        if baseline_reused:
            reporter.write(f"Reusing baseline checkpoint: {checkpoint}")
        else:
            _run(
                [
                    sys.executable,
                    "-u",
                    "src/net_complexity/train.py",
                    "experiment=best_practice_resnet50_on_cifar10",
                    f"hydra.run.dir={baseline_dir}",
                ],
                reporter,
            )

        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError(
                "Baseline training finished without a non-empty best checkpoint: "
                f"{checkpoint}"
            )
        reporter.write(f"Verified baseline checkpoint: {checkpoint}")
        if not baseline_reused:
            status["completed_steps"].append("baseline_training")
        status["current_step"] = "autopruner_tuning"
        status["updated_at"] = _timestamp()
        _write_status(status_path, status)

        _run(
            [
                sys.executable,
                "-u",
                "src/net_complexity/tune.py",
                "--config-name=tune_autopruner_resnet50_cifar10_v100_11h",
                f"model.pretrained_checkpoint={checkpoint}",
                f"tuning.repeats_per_trial={repeats_per_ratio}",
            ],
            reporter,
        )
        status["completed_steps"].append("autopruner_tuning")
        status["current_step"] = None
        status["status"] = "completed"
        status["finished_at"] = _timestamp()
        status["updated_at"] = status["finished_at"]
        _write_status(status_path, status)
        reporter.write(f"[{status['finished_at']}] AutoPruner series completed")
    except BaseException as error:
        status["status"] = "failed"
        status["finished_at"] = _timestamp()
        status["updated_at"] = status["finished_at"]
        status["error_type"] = type(error).__name__
        status["error"] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            status["return_code"] = error.returncode
            status["failed_command"] = list(error.cmd)
        _write_status(status_path, status)
        reporter.write(
            f"[{status['finished_at']}] AutoPruner series failed: "
            f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        reporter.close()


def main() -> None:
    args = _parse_args()
    if args.repeats_per_ratio <= 0:
        raise ValueError("--repeats-per-ratio must be positive.")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = (
        args.log_file
        if args.log_file is not None
        else REPO_ROOT / "outputs" / "logs" / f"{run_id}_autopruner_v100_11h.log"
    )
    if not log_path.is_absolute():
        log_path = (REPO_ROOT / log_path).resolve()
    status_path = log_path.with_suffix(".status.json")
    baseline_checkpoint = args.baseline_checkpoint
    if baseline_checkpoint is not None:
        if not baseline_checkpoint.is_absolute():
            baseline_checkpoint = (REPO_ROOT / baseline_checkpoint).resolve()
        if not baseline_checkpoint.is_file() or baseline_checkpoint.stat().st_size == 0:
            raise FileNotFoundError(
                "--baseline-checkpoint must point to a non-empty file: "
                f"{baseline_checkpoint}"
            )

    if args.detach:
        _start_detached(
            repeats_per_ratio=args.repeats_per_ratio,
            run_id=run_id,
            log_path=log_path,
            status_path=status_path,
            baseline_checkpoint=baseline_checkpoint,
        )
        return

    _claim_active_run(run_id=run_id, pid=os.getpid())
    try:
        _run_series(
            repeats_per_ratio=args.repeats_per_ratio,
            run_id=run_id,
            log_path=log_path,
            status_path=status_path,
            mirror_to_console=not args.no_console_mirror,
            baseline_checkpoint=baseline_checkpoint,
        )
    finally:
        _release_active_run(run_id=run_id)


if __name__ == "__main__":
    main()
