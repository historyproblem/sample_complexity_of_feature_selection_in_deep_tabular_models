"""Console-only one-shot progress; no tensor operations or training state writes."""
from contextlib import contextmanager
from datetime import datetime
import math
import threading
import time


def progress_message(stage, message):
    """Flush immediately, including when output is redirected to a log file."""
    try:
        print(f"[{datetime.now():%H:%M:%S}] [{stage}] {message}", flush=True)
    except (OSError, ValueError):
        # A closed terminal must not cancel an otherwise valid training run.
        pass


@contextmanager
def phase_progress(stage, message, *, interval_seconds=15, ledger=None, total_epochs=None):
    """Show elapsed time and existing CPU ledger counters while a phase runs.

    Reading counters does not synchronize CUDA, sample RNGs, inspect model
    tensors or consume dataloader batches. Epoch counts mean completed epochs;
    updates/examples can advance while an epoch is still running.
    """
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("Progress interval must be finite and positive")
    started = time.monotonic()
    stopped = threading.Event()
    offset = int(ledger.get("global_training_epoch", 0)) if ledger is not None else 0

    def status():
        text = f"elapsed={time.monotonic() - started:.1f}s"
        if ledger is not None:
            completed = int(ledger.get("global_training_epoch", 0)) - offset
            epochs = f"{completed}/{total_epochs}" if total_epochs is not None else str(completed)
            text += (f" | completed_stage_epochs={epochs}"
                     f" | optimizer_updates={ledger.get('optimizer_updates', 0)}"
                     f" | train_examples={ledger.get('consumed_training_examples', 0)}")
        return text

    def heartbeat():
        while not stopped.wait(interval_seconds):
            progress_message(stage, f"running: {message} | {status()}")

    progress_message(stage, f"start: {message}")
    worker = threading.Thread(target=heartbeat, name=f"one-shot-progress-{stage}", daemon=True)
    worker.start()
    outcome = "done"
    try:
        yield
    except BaseException:
        outcome = "failed or interrupted"
        raise
    finally:
        stopped.set()
        worker.join(timeout=0.1)
        progress_message(stage, f"{outcome}: {message} | {status()}")


def epoch_progress(stage, epoch, total, valid, ledger, elapsed_seconds):
    progress_message(stage,
        f"epoch {epoch}/{total} completed | valid_accuracy={float(valid['valid_accuracy']):.4%}"
        f" | valid_ce={float(valid['valid_ce_loss']):.6f}"
        f" | optimizer_updates={ledger['optimizer_updates']}"
        f" | train_examples={ledger['consumed_training_examples']}"
        f" | stage_elapsed={elapsed_seconds:.1f}s")
