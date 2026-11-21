"""Bounded disk retention must preserve both feasible and fallback selection."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_pruning_ablations import fixture
import net_complexity.training.one_shot_pruning as runtime


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("reference_accuracy", [0.0, 1.0])
def test_online_cache_preserves_training_selection_and_final_tensors(tmp_path, monkeypatch, reference_accuracy):
    cfg = fixture(tmp_path / "inputs")
    cfg.model.backbone.resnet_block.train_gate_mode = "gumbel_hard"
    cfg.one_shot.branches = ["inherited"]
    Path(cfg.accuracy_guided.reference.history_path).write_text(
        "global_epoch,valid_accuracy\n" + "".join(f"{i},{reference_accuracy}\n" for i in range(1, 6)))
    original = runtime.run_one_shot_pruning(cfg, tmp_path / "all", keep_epoch_checkpoints=False)
    training = runtime.run_training
    observed = []
    def checked(config, **kwargs):
        callback = kwargs["epoch_end_callback"]
        def bounded(epoch, train, valid, model, optimizer, history):
            before = list(history.checkpoints_dir.glob("epoch_*.pt"))
            assert len(before) <= 2
            callback(epoch, train, valid, model, optimizer, history)
            after = list(history.checkpoints_dir.glob("epoch_*.pt"))
            assert len(after) <= 1
            observed.append((config.run_history.run_name, epoch, len(after)))
        kwargs["epoch_end_callback"] = bounded
        return training(config, **kwargs)
    monkeypatch.setattr(runtime, "run_training", checked)
    limited = runtime.run_one_shot_pruning(cfg, tmp_path / "bounded", keep_epoch_checkpoints=False,
                                          bounded_checkpoints=True)
    for key in ("selected_epoch", "mask_hash", "selected_model_state_hash", "policy"):
        assert original["selection"][key] == limited["selection"][key]
    a, b = original["branches"]["inherited"], limited["branches"]["inherited"]
    for key in ("model_state_hash", "selected_final_epoch", "validation", "quality_feasible"):
        assert a[key] == b[key]
    assert a["ledger"] == b["ledger"]
    assert original["selection"]["trace"]["fallback_used"] == limited["selection"]["trace"]["fallback_used"]
    assert len(limited["selection"]["trace"]["trace"]) == 2
    assert len(observed) == 5 and observed[0][2] == 0  # search epoch1 is outside the candidate window
    search_files = list((tmp_path / "bounded/shared_search").rglob("epoch_*.pt"))
    recovery_files = list((tmp_path / "bounded/inherited").rglob("epoch_*.pt"))
    assert len(search_files) == 1 and not recovery_files
    assert (tmp_path / "bounded/selected_checkpoint.pt").is_file()
    assert (tmp_path / "bounded/inherited/deployment.pt").is_file()
    metrics = list((tmp_path / "bounded").rglob("selection_candidates.jsonl"))
    assert len(metrics) == 2


def test_bounded_cache_rejects_archival_pause_before_training(tmp_path):
    cfg = fixture(tmp_path / "inputs")
    with pytest.raises(ValueError, match="does not support"):
        runtime.run_one_shot_pruning(cfg, tmp_path / "absent", search_only=True,
                                    bounded_checkpoints=True, stop_after_search_epoch=1)
    assert not (tmp_path / "absent").exists()
