"""Cooperative stop at a completed optimizer step; never label a partial epoch complete."""
from __future__ import annotations

import signal
from contextlib import contextmanager

_stop_requested = False


class TrainingInterrupted(RuntimeError):
    def __init__(self, epoch=0, batches=0):
        super().__init__("Training interrupted; run is incomplete.")
        self.epoch = epoch
        self.batches = batches


def check_stop(*, epoch=0, batches=0):
    if _stop_requested:
        raise TrainingInterrupted(epoch, batches)


@contextmanager
def cooperative_signals():
    global _stop_requested
    _stop_requested = False

    def request_stop(_signum, _frame):
        global _stop_requested
        _stop_requested = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        _stop_requested = False
