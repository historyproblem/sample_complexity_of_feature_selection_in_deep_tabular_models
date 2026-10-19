"""Train-only BN calibration for physical search-checkpoint selection."""
from copy import deepcopy
import random

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf, open_dict
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from net_complexity.models.pruning_budget import gates, select_learned_closed
from net_complexity.training import one_shot_pruning as runtime
from net_complexity.training import one_shot_pruning_config as schema
from net_complexity.training.pruning_audit import build_structural
from net_complexity.training.pruning_measurement import (
    calibrate_bn, evaluate_deployment, isolated_diagnostic_rng, state_hash,
)
from net_complexity.training.pruning_synthetic import make_synthetic_config


@pytest.fixture(autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def short_config(directory, *, batches=2):
    cfg = make_synthetic_config(directory, learned_closed=True)
    OmegaConf.update(cfg, "one_shot", {
        "protocol": schema.PROTOCOL,
        "search_epochs": 3,
        "final_epochs": 2,
        "search_selection_last_epochs": 2,
        "search_selection_bn_calibration_batches": batches,
        "branches": ["inherited", "scratch"],
        "scratch_initialization": "pytorch_default_all_trainable_and_bn",
        "inherited_optimizer_state": "mapped_adamw_moments_and_step",
    }, force_add=True)
    cfg.accuracy_guided.total_epochs = cfg.training_arguments.num_epochs = 5
    cfg.accuracy_guided.guard.train_bn_calibration_batches = 0
    cfg.accuracy_guided.stage_plan = [
        dict(id="shared_search", kind="search", epochs=3, commit_allowed=False,
             selected_checkpoint=None, restart_policy="adamw_cosine_restart"),
        dict(id="export_only", kind="commit", epochs=0, commit_allowed=True,
             selected_checkpoint=None, restart_policy="none"),
        dict(id="final_recovery", kind="recovery", epochs=2, commit_allowed=False,
             selected_checkpoint=None, restart_policy="adamw_cosine_restart"),
    ]
    return cfg


class RandomTrainAugmentation(Dataset):
    """Exercise all three parent-process RNG streams during calibration."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        noise = random.random() + float(np.random.random()) + float(torch.rand(()))
        return image + noise * .1, target


@pytest.fixture
def candidate(tmp_path):
    cfg = short_config(tmp_path / "inputs")
    model = instantiate(cfg.model)
    initial = torch.load(cfg.accuracy_guided.initializer.path,
                         map_location="cpu", weights_only=True)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    # Force the width floor to preserve at least one closed channel, while
    # deliberately stale inherited BN statistics require a real reset/pass.
    with torch.no_grad():
        first_gate = next(iter(gates(model).values()))
        first_gate.logits[:, 0].zero_()
        first_gate.logits[:, 1].fill_(-3)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.fill_(20)
                module.running_var.fill_(.01)
                module.num_batches_tracked.fill_(97)
    data = instantiate(cfg.dataloaders, include_test=False)
    train_loader = DataLoader(
        RandomTrainAugmentation(data.train_dataloader.dataset), batch_size=4,
        shuffle=True, generator=torch.Generator().manual_seed(431),
    )
    checkpoint = {"epoch": 2, "model_state_dict": deepcopy(model.state_dict()),
                  "metrics": {"valid_accuracy": .321, "valid_ce_loss": .789}}
    return cfg, model, checkpoint, tmp_path / "epoch_0002.pt", train_loader, data.valid_dataloader


def rng_snapshot(*loaders):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "loaders": [loader.generator.get_state().clone() for loader in loaders],
    }


def assert_same_rng(before, after):
    assert before["python"] == after["python"]
    assert before["numpy"][0] == after["numpy"][0]
    np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
    assert before["numpy"][2:] == after["numpy"][2:]
    assert torch.equal(before["torch"], after["torch"])
    assert len(before["loaders"]) == len(after["loaders"])
    for first, second in zip(before["loaders"], after["loaders"]):
        assert torch.equal(first, second)


def evaluate_candidate(candidate):
    cfg, model, checkpoint, path, train_loader, valid_loader = candidate
    return runtime.evaluate_search_candidate(
        cfg, model, checkpoint, path, train_loader, valid_loader, "cpu", 42,
    )


def test_candidate_score_is_independent_physical_train_bn_then_frozen_validation(candidate):
    cfg, model, _, _, train_loader, valid_loader = candidate
    mask, _ = select_learned_closed(model, {}, float(cfg.accuracy_guided.eligibility.min_keep_ratio))
    with isolated_diagnostic_rng(train_loader, valid_loader):
        expected = build_structural(schema.to_v3_config(cfg), model, mask)
        parameter_hash = state_hash(dict(expected.named_parameters()))
        assert calibrate_bn(expected, train_loader, "cpu", 2, 42) == 2
        expected_metrics = evaluate_deployment(expected, valid_loader, "cpu")
    assert parameter_hash == state_hash(dict(expected.named_parameters()))
    expected_hash = state_hash(expected.state_dict())
    assert all(int(module.num_batches_tracked) == 2 for module in expected.modules()
               if isinstance(module, nn.BatchNorm2d))

    result = evaluate_candidate(candidate)

    assert result["selection_predictor"] == "physical_after_train_bn_calibration"
    assert result["accuracy"] == expected_metrics["accuracy"]
    assert result["ce_loss"] == expected_metrics["ce_loss"]
    assert result["pruning_mask"] == mask
    assert result["raw_gated_validation"] == {"accuracy": .321, "ce_loss": .789}
    calibration = result["selection_bn_calibration"]
    assert calibration["requested_batches"] == calibration["batches"] == 2
    assert calibration["seed"] == 42 and calibration["forward_examples"] == 8
    assert calibration["trainable_weights_unchanged"] is True
    assert calibration["model_state_hash_after"] == expected_hash
    assert calibration["model_state_hash_before"] != calibration["model_state_hash_after"]


def test_candidate_does_not_change_source_checkpoint_weights_modes_or_rng(candidate):
    _, model, checkpoint, _, train_loader, valid_loader = candidate
    source_hash = state_hash(model.state_dict())
    checkpoint_hash = state_hash(checkpoint["model_state_dict"])
    modes = {name: module.training for name, module in model.named_modules()}
    before = rng_snapshot(train_loader, valid_loader)

    evaluate_candidate(candidate)

    assert state_hash(model.state_dict()) == source_hash
    assert state_hash(checkpoint["model_state_dict"]) == checkpoint_hash
    assert {name: module.training for name, module in model.named_modules()} == modes
    assert all(parameter.grad is None for parameter in model.parameters())
    assert_same_rng(before, rng_snapshot(train_loader, valid_loader))


def test_repeat_candidate_reuses_identical_examples_augmentation_and_bn_stats(candidate):
    first = evaluate_candidate(candidate)
    random.random()
    np.random.random(17)
    torch.rand(19)
    second = evaluate_candidate(candidate)
    assert (first["accuracy"], first["ce_loss"]) == (second["accuracy"], second["ce_loss"])
    for key in ("model_state_hash_after", "batches", "forward_examples", "seed"):
        assert first["selection_bn_calibration"][key] == second["selection_bn_calibration"][key]


def test_bn_candidate_rejects_persistent_workers_before_iteration(candidate):
    cfg, model, checkpoint, path, train_loader, valid_loader = candidate
    unsafe_loader = DataLoader(train_loader.dataset, batch_size=4, num_workers=1,
                             persistent_workers=True)
    with pytest.raises(ValueError, match="persistent_workers"):
        runtime.evaluate_search_candidate(
            cfg, model, checkpoint, path, unsafe_loader, valid_loader, "cpu", 42,
        )


def test_missing_option_keeps_legacy_uncalibrated_selection(candidate, monkeypatch):
    cfg, model, _, _, _, valid_loader = candidate
    with open_dict(cfg.one_shot):
        del cfg.one_shot["search_selection_bn_calibration_batches"]
    mask, _ = select_learned_closed(model, {}, float(cfg.accuracy_guided.eligibility.min_keep_ratio))
    with isolated_diagnostic_rng(valid_loader):
        expected = evaluate_deployment(
            runtime.materialize_physical_survivor_carrier(model, mask), valid_loader, "cpu",
        )
    monkeypatch.setattr(runtime, "calibrate_bn", lambda *args, **kwargs: pytest.fail("legacy BN pass"))

    result = evaluate_candidate(candidate)

    assert schema.search_selection_bn_batches(cfg) == 0
    assert result["selection_predictor"] == "dependency_safe_all_physical_survivors_open"
    assert (result["accuracy"], result["ce_loss"]) == (expected["accuracy"], expected["ce_loss"])


@pytest.mark.parametrize("phase", ["search", "recovery"])
@pytest.mark.parametrize("value", [-1, True, None, 1.5, "200"])
def test_candidate_bn_config_refuses_non_integer_or_negative_batch_count(phase, value):
    cfg = schema.compose_config(f"experiment/pruning_v3/resnet50_cifar10_ls0_curve7_{phase}")
    OmegaConf.update(cfg, "one_shot.search_selection_bn_calibration_batches", value, force_add=True)
    with pytest.raises(ValueError, match="search_selection_bn_calibration_batches"):
        schema.validate_config(cfg)


def test_candidate_bn_requires_explicit_checkpoint_window():
    cfg = schema.compose_config()
    OmegaConf.update(cfg, "one_shot.search_selection_bn_calibration_batches", 200, force_add=True)
    with pytest.raises(ValueError, match="search_selection_last_epochs"):
        schema.validate_config(cfg)


@pytest.mark.parametrize("source_batches,recovery_batches", [(200, 0), (0, 200), (200, 100)])
def test_recovery_refuses_different_selection_bn_policy(source_batches, recovery_batches):
    source = schema.compose_config("experiment/pruning_v3/resnet50_cifar10_ls0_curve7_search")
    recovery = schema.compose_config("experiment/pruning_v3/resnet50_cifar10_ls0_curve7_recovery")
    for cfg, batches in ((source, source_batches), (recovery, recovery_batches)):
        OmegaConf.update(cfg, "one_shot.search_selection_bn_calibration_batches", batches, force_add=True)
        schema.validate_config(cfg)
    with pytest.raises(ValueError, match="search_selection_bn_calibration_batches"):
        schema.validate_search_handoff(recovery, source, 31)


def test_recovery_accepts_identical_selection_bn_policy():
    source = schema.compose_config("experiment/pruning_v3/resnet50_cifar10_ls0_curve7_search")
    recovery = schema.compose_config("experiment/pruning_v3/resnet50_cifar10_ls0_curve7_recovery")
    for cfg in (source, recovery):
        OmegaConf.update(cfg, "one_shot.search_selection_bn_calibration_batches", 200, force_add=True)
        assert schema.validate_config(cfg) == 150
    schema.validate_search_handoff(recovery, source, 31)
    schema.validate_search_handoff(recovery, source, 60)


def test_runtime_calibrates_only_in_window_candidates_without_extra_training(tmp_path, monkeypatch):
    cfg = short_config(tmp_path / "inputs")
    checked = []
    original = runtime.evaluate_search_candidate

    def record_candidate(config, model, checkpoint, *args, **kwargs):
        checked.append(int(checkpoint["epoch"]))
        return original(config, model, checkpoint, *args, **kwargs)

    monkeypatch.setattr(runtime, "evaluate_search_candidate", record_candidate)
    result = runtime.run_one_shot_pruning(cfg, tmp_path / "search", search_only=True)

    assert checked == [2, 3]
    assert result["selection"]["selected_epoch"] in checked
    assert result["shared_search_ledger"]["search_epochs_consumed"] == 3
    assert result["shared_search_ledger"]["global_training_epoch"] == 3
    assert result["test_evaluated"] is False
    assert result["branches"] == {}


def test_recovery_reuses_post_bn_selection_score_without_reselecting_or_retraining_search(
    tmp_path, monkeypatch,
):
    cfg = short_config(tmp_path / "inputs")
    search_dir = tmp_path / "search"
    search = runtime.run_one_shot_pruning(cfg, search_dir, search_only=True)
    selected_validation = deepcopy(search["selection"]["selection_validation"])
    selected_calibration = deepcopy(search["selection"]["selection_bn_calibration"])
    monkeypatch.setattr(
        runtime, "evaluate_search_candidate",
        lambda *args, **kwargs: pytest.fail("reused search was evaluated again"),
    )

    recovered = runtime.run_one_shot_pruning(
        cfg, tmp_path / "recovery", reuse_search_from=search_dir,
    )

    assert recovered["selection"]["selection_validation"] == selected_validation
    assert recovered["selection"]["selection_bn_calibration"] == selected_calibration
    assert recovered["selection"]["selected_epoch"] == search["selection"]["selected_epoch"]
    assert recovered["shared_search_ledger"] == search["shared_search_ledger"]
    assert all(branch["ledger"]["global_training_epoch"] == 5
               for branch in recovered["branches"].values())
    assert recovered["test_evaluated"] is False
