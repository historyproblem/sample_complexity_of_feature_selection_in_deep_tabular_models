"""Optimizer-state migration across structural Bottleneck channel pruning."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

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
        "policy": "mapped_adamw_moments_and_step",
        "parameter_tensors_transferred": len(mapped_target_parameters),
        "parameter_state_tensors_transferred": mapped_state_tensors,
        "parameter_state_elements_transferred": mapped_state_elements,
        "source_gate_parameter_states_discarded": len(discarded_names),
        "source_gate_parameter_names_discarded": discarded_names,
        "source_step_min": min(steps),
        "source_step_max": max(steps),
        "scheduler_state_transferred": False,
        "scheduler_policy": "restart_for_compact_stage",
        "parameter_group_hyperparameters_transferred": False,
    }
