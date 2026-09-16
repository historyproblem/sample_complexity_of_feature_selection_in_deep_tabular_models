"""Optimizer-state migration across structural Bottleneck channel pruning."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import torch
import torch.nn as nn

from net_complexity.models.channel_pruning import (
    ParameterTensorMapping,
    build_gated_to_structural_parameter_mappings,
)


_ADAM_PARAMETER_STATE_KEYS = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> set[nn.Parameter]:
    return {
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _mapped_state_value(
    mapping: ParameterTensorMapping,
    key: str,
    value: Any,
) -> Any:
    if not isinstance(value, torch.Tensor):
        return deepcopy(value)
    if key in _ADAM_PARAMETER_STATE_KEYS:
        selected = mapping.select_source_tensor(value.detach())
        return selected.clone().to(device=mapping.target_parameter.device)
    if value.ndim == 0 or value.numel() == 1:
        # Adam's step tensor is intentionally left on the source device.  For
        # ordinary non-capturable AdamW this is CPU even when parameters live
        # on CUDA; copying it to the parameter device changes optimizer mode.
        return value.detach().clone()
    raise ValueError(
        f"Unsupported optimizer state tensor {key!r} for {mapping.source_name}: "
        f"shape={tuple(value.shape)}."
    )


def transfer_adamw_state_to_structural(
    gated_model: nn.Module,
    gated_optimizer: torch.optim.Optimizer,
    structural_model: nn.Module,
    structural_optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Map AdamW moments/steps from a gated model to its compact counterpart.

    A fresh target optimizer is required.  Parameter-group hyperparameters and
    the LR scheduler deliberately remain those of the compact training stage;
    only per-parameter accumulated state is inherited.
    """
    if not isinstance(gated_optimizer, torch.optim.AdamW):
        raise TypeError(
            f"Optimizer handoff requires AdamW, got {type(gated_optimizer).__name__}."
        )
    if not isinstance(structural_optimizer, torch.optim.AdamW):
        raise TypeError(
            f"Structural optimizer handoff requires AdamW, got "
            f"{type(structural_optimizer).__name__}."
        )
    if structural_optimizer.state:
        raise ValueError("Structural optimizer must have empty state before handoff.")

    mappings = build_gated_to_structural_parameter_mappings(gated_model, structural_model)
    source_optimizer_parameters = _optimizer_parameters(gated_optimizer)
    target_optimizer_parameters = _optimizer_parameters(structural_optimizer)
    source_parameter_names = {
        parameter: name for name, parameter in gated_model.named_parameters()
    }
    mapped_source_parameters: set[nn.Parameter] = set()
    mapped_target_parameters: set[nn.Parameter] = set()
    mapped_state_tensors = 0
    mapped_state_elements = 0
    steps: list[int] = []

    for mapping in mappings:
        source_parameter = mapping.source_parameter
        target_parameter = mapping.target_parameter
        if source_parameter not in source_optimizer_parameters:
            raise ValueError(
                f"Source optimizer does not own mapped parameter {mapping.source_name}."
            )
        if target_parameter not in target_optimizer_parameters:
            raise ValueError(
                f"Structural optimizer does not own mapped parameter {mapping.target_name}."
            )
        source_state = gated_optimizer.state.get(source_parameter)
        if not source_state:
            raise ValueError(
                f"Source AdamW checkpoint has no accumulated state for {mapping.source_name}."
            )
        missing = {"step", "exp_avg", "exp_avg_sq"} - set(source_state)
        if missing:
            raise ValueError(
                f"Source AdamW state for {mapping.source_name} is missing {sorted(missing)}."
            )

        target_state = {
            key: _mapped_state_value(mapping, key, value)
            for key, value in source_state.items()
        }
        for key in _ADAM_PARAMETER_STATE_KEYS & set(target_state):
            state_tensor = target_state[key]
            if tuple(state_tensor.shape) != tuple(target_parameter.shape):
                raise AssertionError(
                    f"Mapped {key} shape for {mapping.target_name} differs from its parameter."
                )
            mapped_state_tensors += 1
            mapped_state_elements += int(state_tensor.numel())
        step = target_state["step"]
        step_value = int(step.item()) if isinstance(step, torch.Tensor) else int(step)
        if step_value < 0:
            raise ValueError(f"Negative AdamW step for {mapping.source_name}.")
        steps.append(step_value)
        structural_optimizer.state[target_parameter] = target_state
        mapped_source_parameters.add(source_parameter)
        mapped_target_parameters.add(target_parameter)

    if mapped_target_parameters != target_optimizer_parameters:
        missing_names = sorted(
            name
            for name, parameter in structural_model.named_parameters()
            if parameter in target_optimizer_parameters and parameter not in mapped_target_parameters
        )
        raise AssertionError(f"Optimizer handoff missed compact parameters: {missing_names[:5]}.")

    discarded_names = sorted(
        source_parameter_names.get(parameter, "<unnamed>")
        for parameter in gated_optimizer.state
        if parameter not in mapped_source_parameters
    )
    if any("gumbel_layer" not in name for name in discarded_names):
        raise ValueError(
            "Optimizer handoff would discard non-gate state: "
            f"{[name for name in discarded_names if 'gumbel_layer' not in name][:5]}."
        )

    return {
        "scope": "optimizer_parameter_state_only",
        "policy": "mapped_adamw_moments_and_step",
        "parameter_tensors_transferred": len(mapped_target_parameters),
        "parameter_state_tensors_transferred": mapped_state_tensors,
        "parameter_state_elements_transferred": mapped_state_elements,
        "source_gate_parameter_states_discarded": len(discarded_names),
        "source_gate_parameter_names_discarded": discarded_names,
        "source_step_min": min(steps),
        "source_step_max": max(steps),
        "parameter_group_hyperparameters_transferred": False,
    }


def transfer_cosine_scheduler_state(
    source_state_dict: Mapping[str, Any],
    source_step_count: int,
    target_scheduler_state: Any,
    *,
    source_group_index: int = 0,
) -> dict[str, Any]:
    """Continue the source base-group cosine on a compact one-group optimizer.

    Search has a second optimizer group for gate parameters, while the compact
    model has no gates.  Scheduler lists therefore cannot be loaded verbatim:
    the base group's initial/current learning rates are selected explicitly and
    the gate group's entries are discarded.
    """
    if target_scheduler_state is None:
        raise ValueError("Scheduler handoff requires an enabled target scheduler.")
    scheduler = target_scheduler_state.scheduler
    if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        raise TypeError(
            "Scheduler handoff requires CosineAnnealingLR, got "
            f"{type(scheduler).__name__}."
        )
    source = deepcopy(dict(source_state_dict))
    required = {"T_max", "eta_min", "base_lrs", "last_epoch", "_last_lr"}
    missing = required - set(source)
    if missing:
        raise ValueError(f"Source cosine state is missing {sorted(missing)}.")
    base_lrs = list(source["base_lrs"])
    last_lrs = list(source["_last_lr"])
    if len(base_lrs) != len(last_lrs) or not 0 <= source_group_index < len(base_lrs):
        raise ValueError("Source cosine learning-rate groups are inconsistent.")
    target_groups = scheduler.optimizer.param_groups
    if len(target_groups) != 1:
        raise ValueError(
            "Compact cosine handoff expects exactly one non-gate optimizer group."
        )
    if int(source["T_max"]) != int(scheduler.T_max):
        raise ValueError(
            f"Cosine horizon differs: source T_max={source['T_max']}, "
            f"target T_max={scheduler.T_max}."
        )
    if float(source["eta_min"]) != float(scheduler.eta_min):
        raise ValueError("Source and target cosine eta_min differ.")
    base_lr = float(base_lrs[source_group_index])
    current_lr = float(last_lrs[source_group_index])
    if abs(float(scheduler.base_lrs[0]) - base_lr) > 1e-15:
        raise ValueError("Source and target base learning rates differ.")
    if int(source["last_epoch"]) != int(source_step_count):
        raise ValueError(
            "Source cosine last_epoch and recorded scheduler_step_count differ."
        )

    source["base_lrs"] = [base_lr]
    source["_last_lr"] = [current_lr]
    scheduler.load_state_dict(source)
    target_groups[0]["initial_lr"] = base_lr
    target_groups[0]["lr"] = current_lr
    target_scheduler_state.step_count = int(source_step_count)
    return {
        "policy": "continue_selected_checkpoint_cosine",
        "source_group_index": int(source_group_index),
        "source_group_count": len(base_lrs),
        "target_group_count": len(target_groups),
        "gate_scheduler_group_discarded": len(base_lrs) > len(target_groups),
        "T_max": int(scheduler.T_max),
        "eta_min": float(scheduler.eta_min),
        "last_epoch": int(scheduler.last_epoch),
        "step_count": int(target_scheduler_state.step_count),
        "base_lr": base_lr,
        "current_lr": current_lr,
        "optimizer_group_lr_updated": True,
    }
