from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn

from net_complexity.models.feature_selection import get_gumbel_modules


def _sanitize_name(name: str) -> str:
    return name.replace(".", "_") if name else "root"


class GradientNormLogger:
    """Accumulate per-batch L2 gradient norms and report epoch mean/max values."""

    COMPONENTS = ("ce", "regularization", "total")

    def __init__(
        self,
        model: nn.Module,
        *,
        log_per_layer: bool = True,
        every_n_batches: int = 1,
    ):
        if every_n_batches < 1:
            raise ValueError("every_n_batches must be >= 1.")
        self.log_per_layer = bool(log_per_layer)
        self.every_n_batches = int(every_n_batches)
        self.parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.parameter_layers = self._parameter_layer_names(model)
        self.gumbel_parameter_ids = {
            id(module.logits)
            for module in get_gumbel_modules(model).values()
            if module.logits.requires_grad
        }
        self._values: dict[str, dict[str, list[float]]] = {
            component: defaultdict(list) for component in self.COMPONENTS
        }

    def _parameter_layer_names(self, model: nn.Module) -> dict[int, str]:
        result: dict[int, str] = {}
        for module_name, module in model.named_modules():
            for parameter in module.parameters(recurse=False):
                if parameter.requires_grad:
                    result[id(parameter)] = module_name
        return result

    def collect_autograd(self, component: str, loss: torch.Tensor) -> None:
        gradients = torch.autograd.grad(
            loss,
            self.parameters,
            retain_graph=True,
            allow_unused=True,
        )
        self._collect(component, gradients)

    def collect_total(self) -> None:
        self._collect("total", [parameter.grad for parameter in self.parameters])

    def _collect(self, component: str, gradients) -> None:
        layer_sq_norms: dict[str, float] = {
            layer_name: 0.0 for layer_name in set(self.parameter_layers.values())
        }
        total_sq_norm = 0.0
        gumbel_sq_norm = 0.0

        for parameter, gradient in zip(self.parameters, gradients, strict=True):
            if gradient is None:
                continue
            sq_norm = float(gradient.detach().float().pow(2).sum().item())
            total_sq_norm += sq_norm
            layer_name = self.parameter_layers.get(id(parameter), "root")
            layer_sq_norms[layer_name] = layer_sq_norms.get(layer_name, 0.0) + sq_norm
            if id(parameter) in self.gumbel_parameter_ids:
                gumbel_sq_norm += sq_norm

        values = self._values[component]
        values["total"].append(math.sqrt(total_sq_norm))
        values["gumbel_logits_total"].append(math.sqrt(gumbel_sq_norm))
        if self.log_per_layer:
            for layer_name, sq_norm in layer_sq_norms.items():
                values[f"layer_{_sanitize_name(layer_name)}"].append(math.sqrt(sq_norm))

    def compute(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for component, named_values in self._values.items():
            for name, values in named_values.items():
                if not values:
                    continue
                prefix = f"grad_norm_{component}_{name}"
                metrics[f"{prefix}_mean"] = float(sum(values) / len(values))
                metrics[f"{prefix}_max"] = float(max(values))
        return metrics
