"""Exact parameter budgeting in original channel coordinates (Bottleneck only)."""
from __future__ import annotations

import math
import random
from copy import deepcopy

import torch

from .feature_selection import MaskedGumbelBottleneckLayer, MaskedGumbelLayer

BOUNDARIES = {"gumbel_layer": 2, "mid1_gumbel_layer": 0, "mid2_gumbel_layer": 1}


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


class PhysicalBudget:
    def __init__(self, model, mask):
        validate_mask(model, mask)
        self.blocks = {}
        gate_params = {id(p) for gate in gates(model).values() for p in gate.parameters()}
        self.fixed = sum(p.numel() for p in model.parameters() if id(p) not in gate_params)
        for name, block in model.named_modules():
            if not isinstance(block, MaskedGumbelBottleneckLayer):
                continue
            widths = [block.conv1.out_channels, block.conv2.out_channels, block.conv3.out_channels]
            original = list(widths)
            for suffix, axis in BOUNDARIES.items():
                widths[axis] -= len(mask.get(f"{name}.{suffix}", []))
            if any(conv.bias is None or conv.groups != 1
                   for conv in (block.conv1, block.conv2, block.conv3)):
                raise ValueError("Budget formula requires the repository's biased Bottleneck.")
            self.blocks[name] = (block.conv1.in_channels, widths)
            self.fixed -= self.cost(block.conv1.in_channels, original)
        if not self.blocks:
            raise TypeError("No masked Bottleneck blocks found.")

    @staticmethod
    def cost(inputs, widths):
        w1, w2, outputs = widths
        return inputs * w1 + 9 * w1 * w2 + w2 * outputs + 3 * (w1 + w2 + outputs)

    def total(self):
        return self.fixed + sum(self.cost(inputs, widths) for inputs, widths in self.blocks.values())

    def marginal(self, gate):
        name, suffix = gate.rsplit(".", 1)
        inputs, widths = self.blocks[name]
        narrowed = list(widths)
        narrowed[BOUNDARIES[suffix]] -= 1
        return self.cost(inputs, widths) - self.cost(inputs, narrowed)

    def remove(self, gate):
        name, suffix = gate.rsplit(".", 1)
        self.blocks[name][1][BOUNDARIES[suffix]] -= 1


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
