from hydra.utils import instantiate
import torch

from net_complexity.models.channel_pruning import (
    build_gated_to_structural_parameter_mappings,
)
from net_complexity.models.pruning_budget import select_learned_closed
from net_complexity.training.engine import _build_optimizer
from net_complexity.training.engine import SchedulerState
from net_complexity.training.optimizer_handoff import (
    transfer_adamw_state_to_structural,
    transfer_cosine_scheduler_state,
)
from net_complexity.training.pruning_audit import build_structural
from net_complexity.training.pruning_synthetic import make_synthetic_config


def test_adamw_state_uses_the_same_original_coordinate_mapping_as_weights(tmp_path):
    config = make_synthetic_config(tmp_path / "inputs", learned_closed=True)
    gated = instantiate(config.model)
    initializer = torch.load(
        config.accuracy_guided.initializer.path,
        map_location="cpu",
        weights_only=True,
    )
    gated.load_state_dict(initializer["model_state_dict"], strict=True)
    mask, _ = select_learned_closed(gated, {}, .5)
    structural = build_structural(config, gated, mask)
    source_optimizer, _ = _build_optimizer(config, gated)
    target_optimizer, _ = _build_optimizer(config, structural)

    for parameter_index, parameter in enumerate(gated.parameters(), 1):
        values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
        values = values + parameter_index * 1000
        source_optimizer.state[parameter] = {
            "step": torch.tensor(17.0),
            "exp_avg": values.clone(),
            "exp_avg_sq": values.clone() + .25,
            "max_exp_avg_sq": values.clone() + .5,
        }

    mappings = build_gated_to_structural_parameter_mappings(gated, structural)
    report = transfer_adamw_state_to_structural(
        gated,
        source_optimizer,
        structural,
        target_optimizer,
    )

    assert report["policy"] == "mapped_adamw_moments_and_step"
    assert report["parameter_tensors_transferred"] == len(tuple(structural.parameters()))
    assert report["source_step_min"] == report["source_step_max"] == 17
    assert report["source_gate_parameter_states_discarded"] > 0
    assert report["scope"] == "optimizer_parameter_state_only"

    for mapping in mappings:
        source_state = source_optimizer.state[mapping.source_parameter]
        target_state = target_optimizer.state[mapping.target_parameter]
        assert int(target_state["step"]) == 17
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            expected = mapping.select_source_tensor(source_state[key])
            assert torch.equal(target_state[key], expected), (mapping.target_name, key)
            assert target_state[key].data_ptr() != source_state[key].data_ptr()
        # The model weights and optimizer moments must use one authoritative
        # original-coordinate mapping rather than parallel positional guesses.
        assert torch.equal(
            mapping.target_parameter,
            mapping.select_source_tensor(mapping.source_parameter),
        ), mapping.target_name


def test_optimizer_handoff_refuses_non_adamw_state(tmp_path):
    config = make_synthetic_config(tmp_path / "inputs", learned_closed=True)
    gated = instantiate(config.model)
    initializer = torch.load(
        config.accuracy_guided.initializer.path,
        map_location="cpu",
        weights_only=True,
    )
    gated.load_state_dict(initializer["model_state_dict"], strict=True)
    mask, _ = select_learned_closed(gated, {}, .5)
    structural = build_structural(config, gated, mask)

    source = torch.optim.SGD(gated.parameters(), lr=.1, momentum=.9)
    target, _ = _build_optimizer(config, structural)
    try:
        transfer_adamw_state_to_structural(gated, source, structural, target)
    except TypeError as error:
        assert "AdamW" in str(error)
    else:
        raise AssertionError("Non-AdamW optimizer handoff must fail closed.")


def test_cosine_handoff_selects_base_group_and_continues_exactly():
    source_base = torch.nn.Parameter(torch.tensor(1.0))
    source_gate = torch.nn.Parameter(torch.tensor(2.0))
    source_optimizer = torch.optim.AdamW([
        {"params": [source_base], "lr": .001},
        {"params": [source_gate], "lr": .001, "weight_decay": 0.0},
    ])
    source = SchedulerState(
        torch.optim.lr_scheduler.CosineAnnealingLR(source_optimizer, T_max=5)
    )
    for _ in range(2):
        source_optimizer.zero_grad()
        source_base.grad = torch.ones_like(source_base)
        source_gate.grad = torch.ones_like(source_gate)
        source_optimizer.step()
        source.step()

    target_parameter = torch.nn.Parameter(torch.tensor(1.0))
    target_optimizer = torch.optim.AdamW([target_parameter], lr=.001)
    target = SchedulerState(
        torch.optim.lr_scheduler.CosineAnnealingLR(target_optimizer, T_max=5)
    )
    report = transfer_cosine_scheduler_state(
        source.scheduler.state_dict(), source.step_count, target,
    )

    assert report["policy"] == "continue_selected_checkpoint_cosine"
    assert report["source_group_count"] == 2
    assert report["target_group_count"] == 1
    assert report["gate_scheduler_group_discarded"] is True
    assert target.step_count == source.step_count == 2
    assert target.scheduler.last_epoch == source.scheduler.last_epoch == 2
    assert target_optimizer.param_groups[0]["lr"] == source_optimizer.param_groups[0]["lr"]

    source_optimizer.step()
    target_optimizer.step()
    source.step()
    target.step()
    assert target.step_count == source.step_count == 3
    assert target.scheduler.last_epoch == source.scheduler.last_epoch == 3
    assert target_optimizer.param_groups[0]["lr"] == source_optimizer.param_groups[0]["lr"]
