"""One authorized shared-search comparison; tiny CPU fixtures, no CIFAR access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

from net_complexity.models.channel_pruning import build_structurally_pruned_model_from_config
from net_complexity.models.pruning_budget import gates, select_learned_closed
from net_complexity.training.one_shot_pruning import build_physical_branches, run_one_shot_pruning
from net_complexity.training.one_shot_pruning_config import to_v3_config, validate_config
from net_complexity.training.pruning_audit import build_structural
from net_complexity.training.pruning_measurement import isolated_diagnostic_rng, mask_hash, state_hash
from net_complexity.training.pruning_synthetic import make_synthetic_config


@pytest.fixture(autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def one_shot_fixture(directory, *, learned_closed=True, reject_after=None):
    cfg = make_synthetic_config(directory, learned_closed=learned_closed, reject_after=reject_after)
    OmegaConf.update(cfg, "one_shot", {
        "protocol": "pruning_v3_one_shot_60_90", "search_epochs": 3, "final_epochs": 2,
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
    validate_config(cfg)
    return cfg


def selected_fixture(cfg):
    selected = instantiate(cfg.model)
    checkpoint = torch.load(cfg.accuracy_guided.initializer.path, map_location="cpu", weights_only=True)
    selected.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # Every trainable tensor and every BatchNorm state component differs from
    # constructor defaults. This detects accidental inherited stem/fc/BN state
    # in the scratch branch as well as missed surviving-state transfer.
    with torch.no_grad():
        gate_parameters = {id(parameter) for gate in gates(selected).values() for parameter in gate.parameters()}
        for index, parameter in enumerate(selected.parameters(), 1):
            if id(parameter) not in gate_parameters:
                parameter.fill_(.015 + index * .00001)
        for index, module in enumerate(selected.modules(), 1):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.running_mean.fill_(.05 + index * .001)
                module.running_var.fill_(1.2 + index * .01)
                module.num_batches_tracked.fill_(11)
    mask, _ = select_learned_closed(selected, {}, .5)
    identity = {
        "selected_checkpoint_id": "shared_search/epoch_002",
        "selected_checkpoint_hash": hashlib.sha256(b"synthetic_selected_checkpoint").hexdigest(),
        "selected_model_state_hash": state_hash(selected.state_dict()),
        "mask_hash": mask_hash(mask),
    }
    return selected, mask, identity


def test_physical_branches_share_mask_and_architecture_but_scratch_is_wholly_fresh(tmp_path):
    cfg = one_shot_fixture(tmp_path / "inputs")
    selected, mask, identity = selected_fixture(cfg)
    before = state_hash(selected.state_dict())
    inherited, scratch, report = build_physical_branches(cfg, selected, mask, identity)
    assert state_hash(selected.state_dict()) == before
    assert {key: report[key] for key in identity} == identity
    assert report["inherited"]["initialization"] == "selected_surviving_state"
    assert report["scratch"]["initialization"] == "pytorch_default_all_trainable_and_bn"
    assert report["scratch"]["seed"] == int(cfg.seed)
    assert not gates(inherited) and not gates(scratch)
    assert inherited.lambda_coef == scratch.lambda_coef == 0

    # Independent use of the existing slicing/transfer functions checks every
    # stem/classifier/Conv/BN tensor, including buffers and original ids.
    expected_inherited = build_structural(to_v3_config(cfg), selected, mask)
    assert state_hash(inherited.state_dict()) == state_hash(expected_inherited.state_dict())
    assert report["inherited"]["initialization_state_hash"] == state_hash(inherited.state_dict())

    fresh_cfg = deepcopy(to_v3_config(cfg))
    fresh_cfg.model.lambda_coef = 0
    pruning = OmegaConf.create(dict(enabled=True, mode="explicit", structural=True, mask=mask))
    with isolated_diagnostic_rng():
        torch.manual_seed(int(cfg.seed))
        expected_scratch = build_structurally_pruned_model_from_config(fresh_cfg, pruning)
    assert state_hash(scratch.state_dict()) == state_hash(expected_scratch.state_dict())
    assert report["scratch"]["initialization_state_hash"] == state_hash(scratch.state_dict())

    inherited_parameters, scratch_parameters = dict(inherited.named_parameters()), dict(scratch.named_parameters())
    assert inherited_parameters.keys() == scratch_parameters.keys()
    for name, parameter in scratch_parameters.items():
        assert parameter.shape == inherited_parameters[name].shape
        assert not torch.equal(parameter, inherited_parameters[name]), name
    for name, module in scratch.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            inherited_bn = dict(inherited.named_modules())[name]
            assert torch.equal(module.running_mean, torch.zeros_like(module.running_mean))
            assert torch.equal(module.running_var, torch.ones_like(module.running_var))
            assert module.num_batches_tracked == 0
            assert inherited_bn.num_batches_tracked == 11
    assert any("conv1.weight" in name for name in scratch_parameters)
    assert "backbone.fc.weight" in scratch_parameters and "backbone.fc.bias" in scratch_parameters
    for key in inherited.state_dict():
        if key.endswith("active_indices"):
            assert torch.equal(inherited.state_dict()[key], scratch.state_dict()[key]), key

    # Fresh initialization cannot depend on preceding diagnostic/random draws.
    torch.rand(17)
    _, scratch_again, again_report = build_physical_branches(cfg, selected, mask, identity)
    assert state_hash(scratch_again.state_dict()) == state_hash(scratch.state_dict())
    assert again_report["architecture_hash"] == report["architecture_hash"]


@pytest.mark.parametrize("mismatched", ["mask_hash", "selected_model_state_hash"])
def test_physical_branch_fork_rejects_an_identity_from_another_selection(tmp_path, mismatched):
    cfg = one_shot_fixture(tmp_path / "inputs")
    selected, mask, identity = selected_fixture(cfg)
    identity[mismatched] = "0" * 64
    with pytest.raises(ValueError, match="(?i)(hash|identity|selection|checkpoint|mask)"):
        build_physical_branches(cfg, selected, mask, identity)


@pytest.mark.parametrize("learned_closed", [False, True])
def test_one_shared_search_feeds_export_and_both_branches_with_distinct_compute_ledger(tmp_path, learned_closed):
    cfg = one_shot_fixture(tmp_path / "inputs", learned_closed=learned_closed)
    reference_before = Path(cfg.accuracy_guided.reference.history_path).read_bytes()
    initializer_before = Path(cfg.accuracy_guided.initializer.path).read_bytes()
    output = tmp_path / "run"
    result = run_one_shot_pruning(cfg, output)
    assert result["status"] == "completed"
    selection = result["selection"]
    selected_file = output / "selected_checkpoint.pt"
    selected_payload = torch.load(selected_file, map_location="cpu", weights_only=True)
    assert hashlib.sha256(selected_file.read_bytes()).hexdigest() == selection["selected_checkpoint_hash"]
    assert state_hash(selected_payload["model_state_dict"]) == selection["selected_model_state_hash"]
    assert selected_payload["epoch_event"]["alpha_next"] > selected_payload["epoch_event"]["alpha_used"]
    assert selection["mask_hash"] == mask_hash(selection["pruning_mask"])
    assert sum(map(len, selection["pruning_mask"].values())) == (2 if learned_closed else 0)
    selected_steps = {
        int(item["step"])
        for item in selected_payload["optimizer_state_dict"]["state"].values()
    }
    assert len(selected_steps) == 1
    selected_step = selected_steps.pop()
    identities = [result["export_only"], result["branches"]["inherited"], result["branches"]["scratch"]]
    for record in identities:
        for key in ("selected_checkpoint_id", "selected_checkpoint_hash", "selected_model_state_hash", "mask_hash"):
            assert record[key] == selection[key]
    inherited, scratch = result["branches"]["inherited"], result["branches"]["scratch"]
    assert inherited["architecture_hash"] == scratch["architecture_hash"] == result["export_only"]["architecture_hash"]
    for branch in (inherited, scratch):
        assert branch["pruning_mask"] == selection["pruning_mask"]
        assert branch["training_initializer_verified"]
        assert branch["ledger"] == dict(global_training_epoch=5, search_epochs_consumed=3,
                                         optimizer_updates=10, consumed_training_examples=40)
    for name in ("inherited", "scratch"):
        initial_payload = torch.load(output / name / "initial_state.pt", map_location="cpu", weights_only=True)
        final_payload = torch.load(output / name / "deployment.pt", map_location="cpu", weights_only=True)
        for payload in (initial_payload, final_payload):
            for key in ("selected_checkpoint_id", "selected_checkpoint_hash", "selected_model_state_hash", "mask_hash"):
                assert payload[key] == selection[key]
            assert payload["model_state_hash"] == state_hash(payload["model_state_dict"])
            assert payload["pruning_mask"] == selection["pruning_mask"]
        assert initial_payload["model_state_hash"] == result["branches"][name]["initialization_state_hash"]
        assert final_payload["ledger"] == result["branches"][name]["ledger"]
        first_epoch_paths = list((output / name).rglob("epoch_0001.pt"))
        assert len(first_epoch_paths) == 1
        first_epoch = torch.load(first_epoch_paths[0], map_location="cpu", weights_only=True)
        optimizer_state = first_epoch["optimizer_state_dict"]["state"]
        expected_step = selected_step + 2 if name == "inherited" else 2
        assert optimizer_state and all(
            int(item["step"]) == expected_step for item in optimizer_state.values()
        )
        assert first_epoch["scheduler_step_count"] == 1
        assert not any("gumbel_layer" in key for key in first_epoch["model_state_dict"])
        assert first_epoch["metrics"]["train_regularization_loss"] == 0
        assert all(event["train_L_gate_mean"] == 0 for event in result["stages"][name]["epoch_events"])
    assert inherited["optimizer_state_initialization"] == "mapped_adamw_moments_and_step"
    assert inherited["optimizer_handoff"]["source_step_min"] == selected_step
    assert inherited["optimizer_handoff"]["source_step_max"] == selected_step
    assert inherited["optimizer_handoff"]["scheduler_state_transferred"] is False
    assert scratch["optimizer_state_initialization"] == "fresh"
    assert scratch["optimizer_handoff"] is None
    exported = torch.load(output / "export_only/deployment.pt", map_location="cpu", weights_only=True)
    assert exported["model_state_hash"] == inherited["initialization_state_hash"]
    assert exported["model_state_hash"] == state_hash(exported["model_state_dict"])
    assert exported["training_epochs"] == exported["bn_calibration_batches"] == 0
    assert result["export_only"]["weights_and_bn_unchanged"]
    assert len(list((output / "shared_search").rglob("epoch_*.pt"))) == 3
    assert inherited["initialization_state_hash"] != scratch["initialization_state_hash"]
    ledger = result["compute_ledger"]
    assert ledger["shared_search_epochs"] == 3
    assert ledger["branch_final_training_epochs"] == {"inherited": 2, "scratch": 2}
    assert ledger["actual_training_epochs_executed"] == 7
    assert ledger["actual_optimizer_updates"] == 14
    assert ledger["actual_consumed_training_examples"] == 56
    assert Path(cfg.accuracy_guided.reference.history_path).read_bytes() == reference_before
    assert Path(cfg.accuracy_guided.initializer.path).read_bytes() == initializer_before
    with pytest.raises(FileExistsError):
        run_one_shot_pruning(cfg, output)


def test_no_feasible_shared_search_never_starts_a_recovery_branch(tmp_path):
    cfg = one_shot_fixture(tmp_path / "inputs", reject_after=1)
    result = run_one_shot_pruning(cfg, tmp_path / "run")
    assert result["status"] == "no_feasible_search"
    assert result["branches"] == {}
    assert result["export_only"] is None
    assert result["compute_ledger"]["shared_search_epochs"] == 3
    assert result["compute_ledger"]["actual_training_epochs_executed"] == 3


def test_unexplained_actual_gated_export_mismatch_stops_before_branch_training(tmp_path, monkeypatch):
    from net_complexity.training import one_shot_pruning as runtime
    cfg = one_shot_fixture(tmp_path / "inputs", learned_closed=True)
    original = runtime.gated_export_equivalence

    def mismatched_gated_predictor(carrier, physical, *args, **kwargs):
        # Keep the actual transfer model and its all-open equivalence intact;
        # exercise a real output mismatch in the independent gated-predictor
        # check. This must not be treated as an expected blocked-gate jump.
        wrong_predictor = deepcopy(carrier)
        with torch.no_grad():
            wrong_predictor.backbone.fc.bias.add_(.25)
        return original(wrong_predictor, physical, *args, **kwargs)

    monkeypatch.setattr(runtime, "gated_export_equivalence", mismatched_gated_predictor)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="(?i)(gated|export|equivalence)"):
        runtime.run_one_shot_pruning(cfg, output)
    state = json.loads((output / "one_shot_state.json").read_text())
    diagnostic = json.loads((output / "export_only/diagnostics.json").read_text())
    assert state["status"] == "failed"
    assert state["branches"] == {}
    assert state["shared_search_ledger"] == dict(global_training_epoch=3, search_epochs_consumed=3,
                                                optimizer_updates=6, consumed_training_examples=24)
    assert state["compute_ledger"]["actual_training_epochs_executed"] == 3
    assert diagnostic["blocked_closed_survivors_opened_by_export"] == []
    assert diagnostic["gated_equivalent_on_checked_batch"] is False
    assert diagnostic["non_equivalence_reason"] == "gated_export_function_difference"
    assert not (output / "inherited").exists() and not (output / "scratch").exists()


def test_known_floor_blocked_survivor_opening_is_measured_and_both_branches_continue(tmp_path):
    cfg = one_shot_fixture(tmp_path / "inputs", learned_closed=True)
    initial_path = Path(cfg.accuracy_guided.initializer.path)
    payload = torch.load(initial_path, map_location="cpu", weights_only=True)
    for name, tensor in payload["model_state_dict"].items():
        if name.endswith(".logits"):
            tensor[:, 0] = 0
            tensor[:, 1] = -float(torch.log(torch.tensor(9.0)))
    payload["model_state_hash"] = state_hash(payload["model_state_dict"])
    torch.save(payload, initial_path)
    reference_path = Path(cfg.accuracy_guided.reference.state_path)
    reference = json.loads(reference_path.read_text())
    reference["common_init_hash"] = payload["model_state_hash"]
    reference_path.write_text(json.dumps(reference))
    result = run_one_shot_pruning(cfg, tmp_path / "run")
    diagnostic = result["export_only"]
    assert diagnostic["blocked_closed_survivors_opened_by_export"]
    assert diagnostic["gated_equivalent_on_checked_batch"] is False
    assert diagnostic["non_equivalence_reason"] == "blocked_closed_survivors_opened"
    assert diagnostic["transfer_equivalence"]["status"].startswith("passed")
    assert diagnostic["weights_and_bn_unchanged"]
    assert result["status"] == "completed"
    assert set(result["branches"]) == {"inherited", "scratch"}
    assert result["compute_ledger"]["actual_training_epochs_executed"] == 7
