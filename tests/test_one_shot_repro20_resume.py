"""Real CPU search prefixes retain the original horizon and resume exactly."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from net_complexity.training import one_shot_pruning as runtime
from net_complexity.training.one_shot_pruning_config import validate_config
from net_complexity.training.pruning_synthetic import make_synthetic_config


@pytest.fixture(autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def short_config(directory):
    config = make_synthetic_config(directory, learned_closed=True)
    OmegaConf.update(config, "one_shot", {
        "protocol": "pruning_v3_one_shot_60_90",
        "search_epochs": 3,
        "final_epochs": 2,
        "search_selection_last_epochs": 1,
        "branches": ["inherited", "scratch"],
        "scratch_initialization": "pytorch_default_all_trainable_and_bn",
        "inherited_optimizer_state": "mapped_adamw_moments_and_step",
    }, force_add=True)
    config.accuracy_guided.total_epochs = config.training_arguments.num_epochs = 5
    config.accuracy_guided.guard.train_bn_calibration_batches = 0
    config.accuracy_guided.stage_plan = [
        dict(id="shared_search", kind="search", epochs=3, commit_allowed=False,
             selected_checkpoint=None, restart_policy="adamw_cosine_restart"),
        dict(id="export_only", kind="commit", epochs=0, commit_allowed=True,
             selected_checkpoint=None, restart_policy="none"),
        dict(id="final_recovery", kind="recovery", epochs=2, commit_allowed=False,
             selected_checkpoint=None, restart_policy="adamw_cosine_restart"),
    ]
    validate_config(config)
    return config


def checkpoint(state, epoch):
    directory = Path(state["stages"]["shared_search"]["run_dir"]) / "checkpoints"
    return torch.load(directory / f"epoch_{epoch:04d}.pt", map_location="cpu", weights_only=True)


def assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_exact(first, second)
    else:
        assert left == right


def file_hashes(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def test_real_search_prefix_and_resume_match_uninterrupted_training(tmp_path):
    config = short_config(tmp_path / "inputs")
    full = runtime.run_one_shot_pruning(config, tmp_path / "full", search_only=True)
    prefix_root = tmp_path / "prefix"
    prefix = runtime.run_one_shot_pruning(
        config, prefix_root, search_only=True,
        stop_after_search_epoch=2, search_diagnostics=True,
    )

    assert prefix["status"] == "search_paused"
    assert prefix["shared_search_ledger"]["global_training_epoch"] == 2
    assert prefix["shared_search_ledger"]["search_epochs_consumed"] == 2
    assert prefix["compute_ledger"]["actual_training_epochs_executed"] == 2
    assert prefix["per_branch_total_allocated"] == 5
    assert prefix["branches"] == {}
    assert prefix["selection"] is None
    assert prefix["export_only"] is None
    assert prefix["test_evaluated"] is False
    assert not (prefix_root / "selection.json").exists()
    assert not (prefix_root / "export_only").exists()
    stage = prefix["stages"]["shared_search"]
    assert stage["status"] == "paused"
    assert stage["epochs_allocated"] == 3
    assert stage["scheduler"]["T_max"] == 3
    assert stage["training_initializer_verified"] is True
    assert (Path(stage["run_dir"]) / "checkpoints/epoch_0001.pt").is_file()
    trace = [json.loads(line) for line in
             (prefix_root / "search_diagnostics/trace.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in trace if row["kind"] == "epoch_end"] == [1, 2]

    before = file_hashes(prefix_root)
    resumed = runtime.run_one_shot_pruning(
        config, tmp_path / "resumed", search_only=True, resume_search_from=prefix_root,
    )
    assert file_hashes(prefix_root) == before
    assert resumed["status"] == "search_only_completed"
    assert resumed["shared_search_ledger"] == full["shared_search_ledger"]
    assert resumed["compute_ledger"]["actual_training_epochs_executed"] == 1
    assert resumed["compute_ledger"]["attributed_training_epochs_including_reused_search"] == 3
    assert resumed["selection"]["selected_epoch"] == full["selection"]["selected_epoch"] == 3
    assert resumed["test_evaluated"] is False

    for epoch in (1, 2, 3):
        expected, actual = checkpoint(full, epoch), checkpoint(resumed, epoch)
        for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
                    "scheduler_step_count", "consumed_ledger", "rng_state"):
            assert_exact(expected[key], actual[key])
        for key in ("controller_after_feedback", "alpha_used", "alpha_next", "clocks",
                    "eval_runtime", "continuation_runtime"):
            assert_exact(expected["epoch_event"][key], actual["epoch_event"][key])


def test_resume_rejects_changed_search_recipe_before_training(tmp_path, monkeypatch):
    config = short_config(tmp_path / "inputs")
    prefix_root = tmp_path / "prefix"
    runtime.run_one_shot_pruning(
        config, prefix_root, search_only=True, stop_after_search_epoch=2,
    )
    changed = deepcopy(config)
    changed.training_arguments.adaptive_lambda.hard_drop += .01
    monkeypatch.setattr(runtime, "run_training", lambda *args, **kwargs:
                        pytest.fail("Changed recipe started training instead of rejecting resume"))
    with pytest.raises(ValueError, match="(?i)(config|recipe|resume)"):
        runtime.run_one_shot_pruning(
            changed, tmp_path / "changed", search_only=True, resume_search_from=prefix_root,
        )
