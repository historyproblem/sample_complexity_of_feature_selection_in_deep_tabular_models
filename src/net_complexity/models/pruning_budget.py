"""Exact parameter budgeting in original channel coordinates."""
from __future__ import annotations

import math
import random
from copy import deepcopy

import torch

from .feature_selection import MaskedGumbelBottleneckLayer, MaskedGumbelLayer
from .mobilenet_v2 import MaskedGumbelInvertedResidual

BOTTLENECK_BOUNDARIES = {
    "gumbel_layer": 2,
    "mid1_gumbel_layer": 0,
    "mid2_gumbel_layer": 1,
}
MOBILENET_BOUNDARIES = {
    "gumbel_layer": "output",
    "mid_gumbel_layer": "mid",
}


def gates(model):
    return {name: module for name, module in model.named_modules()
            if isinstance(module, MaskedGumbelLayer)}


def validate_mask(model, mask, *, previous=None, min_keep_ratio=0.0):
    if not 0 <= min_keep_ratio <= 1:
        raise ValueError("min_keep_ratio must be in [0, 1].")
    available = gates(model)
    for name, indices in mask.items():
        if name not in available:
            raise ValueError(f"Unknown/disabled gate: {name}")
        width = available[name].logits.shape[0]
        if any(type(i) is not int or not 0 <= i < width for i in indices):
            raise ValueError(f"Invalid original-coordinate indices for {name}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate indices for {name}")
        if width - len(indices) < max(1, math.ceil(width * min_keep_ratio)):
            raise ValueError(f"Original-width floor violated for {name}")
    for name, indices in (previous or {}).items():
        if not set(indices).issubset(mask.get(name, [])):
            raise ValueError(f"Nonmonotone permanent mask at {name}")


def checkpoint_probabilities(model):
    """Read the loaded selected checkpoint, never last-epoch metric columns."""
    result = {}
    for name, gate in gates(model).items():
        probabilities = gate.logits.detach().float().softmax(-1)[:, 1].cpu()
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError(f"Non-finite probabilities in {name}")
        result[name] = probabilities.tolist()
    return result


def select_learned_closed(model, previous, min_keep_ratio=0.0, *, dependency_groups=None):
    """Materialize the maximal valid deterministic learned-closed subset.

    Costs are measurements, never a removal quota. Coordinates are original
    ids. Dependency groups, if supplied, are iterables of (gate_name, id); an
    open survivor blocks its entire connected group. The repository's ResNet
    and MobileNetV2 slicing has independent boundaries and needs no extra groups.
    This function does not mutate either the checkpoint or its permanent mask.
    """
    validate_mask(model, previous, min_keep_ratio=min_keep_ratio)
    available = gates(model)
    records, eligible, runtime_blocked = {}, set(), []
    for name, gate in available.items():
        survivor = gate.get_permanent_survivor_mask()
        disabled = set((~survivor).nonzero().flatten().tolist())
        if disabled != set(previous.get(name, [])):
            raise ValueError(f"Checkpoint permanent mask differs from supplied previous mask at {name}")
        if gate.initial_channels != gate.logits.shape[0]:
            raise ValueError("Learned-closed selection requires the original-coordinate carrier")
        raw, effective = gate.get_raw_selection_probs(), gate.get_effective_selection_probs()
        if not bool(torch.isfinite(raw).all() and torch.isfinite(effective).all()):
            raise FloatingPointError(f"Non-finite checkpoint probabilities in {name}")
        raw_open = gate.get_hard_gate_decisions(raw=True, apply_permanent_mask=False)
        eval_open = gate.get_hard_gate_decisions(apply_permanent_mask=False)
        for index in survivor.nonzero().flatten().tolist():
            key = (name, index)
            records[key] = {"gate": name, "original_id": index,
                            "raw_probability": float(raw[index].item()),
                            "effective_probability": float(effective[index].item()),
                            "threshold": gate.gate_threshold}
            if not bool(raw_open[index]) and not bool(eval_open[index]):
                eligible.add(key)
            elif not bool(raw_open[index]):
                runtime_blocked.append({**records[key], "reason": "raw_closed_effective_open"})

    # Connected components ensure overlapping dependencies cannot cascade an
    # ineligible/open component into a removal.
    components = []
    for group in dependency_groups or []:
        component = set()
        for name, index in group:
            if name not in available or type(index) is not int or not 0 <= index < available[name].initial_channels:
                raise ValueError("Invalid original id in dependency group")
            if index not in previous.get(name, []):
                component.add((name, index))
        merged = []
        for old in components:
            if old & component:
                component |= old
            else:
                merged.append(old)
        components = merged + ([component] if component else [])
    grouped = set().union(*components) if components else set()
    components.extend({key} for key in eligible - grouped)
    components = [group for group in components if group & eligible]
    rank = lambda key: (records[key]["raw_probability"], key[0], key[1])
    components.sort(key=lambda group: min(rank(key) for key in group))

    costs = PhysicalBudget(model, previous)
    before = costs.total()
    mask = deepcopy(previous)
    blocked, removed = [], []
    for group in components:
        reason = None
        if not group.issubset(eligible):
            reason = "dependency_contains_open_survivor"
        else:
            for name in sorted({key[0] for key in group}):
                width = available[name].initial_channels
                count = sum(key[0] == name for key in group)
                if width - len(mask.get(name, [])) - count < max(1, math.ceil(width * min_keep_ratio)):
                    reason = "blocked_by_floor"
                    break
        if reason:
            blocked.extend({**records[key], "reason": reason,
                            "dependency_group": [[name, index] for name, index in sorted(group)]}
                           for key in sorted(group & eligible))
            continue
        for name, index in sorted(group, key=rank):
            costs.remove(name)
            mask.setdefault(name, []).append(index)
            removed.append(records[(name, index)])
    mask = {name: sorted(indices) for name, indices in mask.items() if indices}
    validate_mask(model, mask, previous=previous, min_keep_ratio=min_keep_ratio)
    report = {
        "drop_mode": "learned_closed_gates", "params_before": before,
        "params_after": costs.total(), "removed_params": before - costs.total(),
        "eligible": [records[key] for key in sorted(eligible, key=rank)],
        "blocked": blocked, "removed": removed,
        "runtime_blocked": runtime_blocked,
        "eligible_original_ids": {name: sorted(index for gate, index in eligible if gate == name)
                                  for name in sorted({key[0] for key in eligible})},
        "removed_original_ids": {name: sorted(set(mask.get(name, [])) - set(previous.get(name, [])))
                                 for name in mask if set(mask[name]) - set(previous.get(name, []))},
        "min_keep_ratio": float(min_keep_ratio), "minimum_channels": 1,
        "dependency_policy": "block_if_any_open_component",
        "no_op_reason": (None if removed else "no_new_pruning" if not eligible
                         else "blocked_by_floor" if all(item["reason"] == "blocked_by_floor" for item in blocked)
                         else "blocked_by_dependencies"),
        "learned_candidates_materialized": len(removed),
        "compression_achieved": costs.total() < before,
    }
    return mask, report


class PhysicalBudget:
    def __init__(self, model, mask):
        validate_mask(model, mask)
        self.blocks = {}
        gate_params = {id(p) for gate in gates(model).values() for p in gate.parameters()}
        self.fixed = sum(p.numel() for p in model.parameters() if id(p) not in gate_params)
        architecture = None
        for name, block in model.named_modules():
            if isinstance(block, MaskedGumbelBottleneckLayer):
                if architecture not in (None, "bottleneck"):
                    raise TypeError("A pruning carrier cannot mix Bottleneck and MobileNetV2 blocks.")
                architecture = "bottleneck"
                widths = [
                    block.conv1.out_channels,
                    block.conv2.out_channels,
                    block.conv3.out_channels,
                ]
                original = list(widths)
                for suffix, axis in BOTTLENECK_BOUNDARIES.items():
                    widths[axis] -= len(mask.get(f"{name}.{suffix}", []))
                if any(
                    conv.bias is None or conv.groups != 1
                    for conv in (block.conv1, block.conv2, block.conv3)
                ):
                    raise ValueError("Budget formula requires the repository's biased Bottleneck.")
                self.blocks[name] = {
                    "kind": "bottleneck",
                    "inputs": block.conv1.in_channels,
                    "widths": widths,
                }
                self.fixed -= self._bottleneck_cost(block.conv1.in_channels, original)
            elif isinstance(block, MaskedGumbelInvertedResidual):
                if architecture not in (None, "mobilenetv2"):
                    raise TypeError("A pruning carrier cannot mix Bottleneck and MobileNetV2 blocks.")
                architecture = "mobilenetv2"
                output_width = int(block.oup)
                mid_width = int(block.hidden_dim)
                widths = {
                    "output": output_width
                    - len(mask.get(f"{name}.gumbel_layer", [])),
                    "mid": mid_width
                    - len(mask.get(f"{name}.mid_gumbel_layer", [])),
                }
                self._validate_mobilenet_block(block)
                self.blocks[name] = {
                    "kind": "mobilenetv2",
                    "inputs": int(block.inp),
                    "has_expand": bool(block.has_expand),
                    "widths": widths,
                }
                self.fixed -= self._mobilenet_cost(
                    int(block.inp),
                    mid_width,
                    output_width,
                    has_expand=bool(block.has_expand),
                )
        if not self.blocks:
            raise TypeError("No supported masked Bottleneck or MobileNetV2 blocks found.")

    @staticmethod
    def _bottleneck_cost(inputs, widths):
        w1, w2, outputs = widths
        return inputs * w1 + 9 * w1 * w2 + w2 * outputs + 3 * (w1 + w2 + outputs)

    # Retain the historic public helper used by older analysis code.
    cost = _bottleneck_cost

    @staticmethod
    def _mobilenet_cost(inputs, mid, outputs, *, has_expand):
        # All convolutions are bias-free and every BatchNorm is affine.
        # expand: inputs*mid + 2*mid
        # depthwise: 3*3*mid + 2*mid
        # project: mid*outputs + 2*outputs
        expand = inputs * mid + 2 * mid if has_expand else 0
        return expand + 11 * mid + mid * outputs + 2 * outputs

    @staticmethod
    def _validate_mobilenet_block(block):
        convolutions = [block.branch.depthwise[0], block.branch.project]
        batch_norms = [block.branch.depthwise[1], block.branch.project_bn]
        if block.has_expand:
            convolutions.append(block.branch.expand[0])
            batch_norms.append(block.branch.expand[1])
        if any(conv.bias is not None for conv in convolutions):
            raise ValueError("MobileNetV2 budget requires bias-free convolutions.")
        depthwise = block.branch.depthwise[0]
        if depthwise.groups != block.hidden_dim or depthwise.kernel_size != (3, 3):
            raise ValueError("MobileNetV2 budget requires the repository's 3x3 depthwise convolution.")
        if any(not bn.affine for bn in batch_norms):
            raise ValueError("MobileNetV2 budget requires affine BatchNorm layers.")

    @classmethod
    def _block_cost(cls, state):
        if state["kind"] == "bottleneck":
            return cls._bottleneck_cost(state["inputs"], state["widths"])
        return cls._mobilenet_cost(
            state["inputs"],
            state["widths"]["mid"],
            state["widths"]["output"],
            has_expand=state["has_expand"],
        )

    def total(self):
        return self.fixed + sum(self._block_cost(state) for state in self.blocks.values())

    def marginal(self, gate):
        name, suffix = gate.rsplit(".", 1)
        state = self.blocks[name]
        before = self._block_cost(state)
        narrowed = dict(state)
        if state["kind"] == "bottleneck":
            narrowed["widths"] = list(state["widths"])
            narrowed["widths"][BOTTLENECK_BOUNDARIES[suffix]] -= 1
        else:
            narrowed["widths"] = dict(state["widths"])
            narrowed["widths"][MOBILENET_BOUNDARIES[suffix]] -= 1
        return before - self._block_cost(narrowed)

    def remove(self, gate):
        name, suffix = gate.rsplit(".", 1)
        state = self.blocks[name]
        if state["kind"] == "bottleneck":
            state["widths"][BOTTLENECK_BOUNDARIES[suffix]] -= 1
        else:
            state["widths"][MOBILENET_BOUNDARIES[suffix]] -= 1


def select_by_budget(model, previous, fraction, min_keep_ratio, *, ranking="learned", seed=42):
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("fraction must be finite and in [0, 1).")
    validate_mask(model, previous, min_keep_ratio=min_keep_ratio)
    probabilities = checkpoint_probabilities(model)
    costs = PhysicalBudget(model, previous)
    before = costs.total()
    limit = math.floor(before * fraction)
    candidates = [(prob, name, i) for name, values in probabilities.items()
                  for i, prob in enumerate(values) if i not in set(previous.get(name, []))]
    candidates.sort()
    if ranking == "random":
        candidates.sort(key=lambda item: (item[1], item[2]))
        random.Random(seed).shuffle(candidates)
    elif ranking != "learned":
        raise ValueError("ranking must be learned or random.")
    mask = deepcopy(previous)
    spent = 0
    for _, name, index in candidates:
        width = len(probabilities[name])
        if width - len(mask.get(name, [])) <= max(1, math.ceil(width * min_keep_ratio)):
            continue
        marginal = costs.marginal(name)
        if spent + marginal > limit:
            continue
        costs.remove(name)
        spent += marginal
        mask.setdefault(name, []).append(index)
    mask = {name: sorted(indices) for name, indices in mask.items() if indices}
    validate_mask(model, mask, previous=previous, min_keep_ratio=min_keep_ratio)
    assert before - costs.total() == spent
    return mask, {"params_before": before, "params_after": costs.total(),
                  "budget_params": limit, "removed_params": spent,
                  "unused_budget_params": limit - spent, "ranking": ranking}
