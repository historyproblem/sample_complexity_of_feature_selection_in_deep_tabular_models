from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn

from net_complexity.metrics.base import BaseMetric
from net_complexity.metrics.inspection import BartlettCollector
from net_complexity.metrics.inspection import ComputeCollector
from net_complexity.metrics.inspection import WeightCollector


def _unwrap_model(model: nn.Module | None) -> nn.Module | None:
    if model is None:
        return None

    backbone = getattr(model, "backbone", None)
    if isinstance(backbone, nn.Module):
        return backbone

    return model


def _sanitize_layer_name(name: str) -> str:
    return name.replace(".", "_") if name else "root"


def _numeric_items(prefix: str, values) -> dict[str, int | float]:
    '''Extract numeric items from a dataclass instance, prefixing keys with the given prefix.'''
    result: dict[str, int | float] = {}
    for key, value in asdict(values).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result[f"{prefix}_{key}"] = value
    return result


class WeightStatisticsMetric(BaseMetric):
    def __init__(self, sparsity_eps: float = 1e-4, log_per_layer: bool = False):
        self.collector = WeightCollector(sparsity_eps=sparsity_eps)
        self.log_per_layer = log_per_layer
        self.reset()

    def update(self, input, output, targets, model=None):
        self.model = _unwrap_model(model)

    def compute(self):
        if self.model is None:
            return {}

        per_layer = {}
        for name, module in self.model.named_modules():
            if self.collector.can_handle(module):
                per_layer[name] = self.collector.collect(module)

        if not per_layer:
            return {}

        total_params = sum(layer.num_params for layer in per_layer.values())
        total_l1 = sum(layer.l1_norm for layer in per_layer.values())
        total_l2 = math.sqrt(sum(layer.l2_norm ** 2 for layer in per_layer.values()))
        total_fro = math.sqrt(sum(layer.fro_norm ** 2 for layer in per_layer.values()))
        mean_sparsity = sum(layer.weight_sparsity for layer in per_layer.values()) / len(per_layer)
        weighted_sparsity = sum(
            layer.weight_sparsity * layer.num_params for layer in per_layer.values()
        ) / max(total_params, 1)

        metrics = {
            "weights_num_layers": len(per_layer),
            "weights_total_params": total_params,
            "weights_l1_norm_total": total_l1,
            "weights_l2_norm_total": total_l2,
            "weights_fro_norm_total": total_fro,
            "weights_mean_sparsity": mean_sparsity,
            "weights_weighted_sparsity": weighted_sparsity,
        }

        if self.log_per_layer:
            for name, layer_metrics in per_layer.items():
                metrics.update(
                    _numeric_items(
                        f"weights_layer_{_sanitize_layer_name(name)}",
                        layer_metrics,
                    )
                )

        return metrics

    def reset(self):
        self.model = None


class BartlettSpectralComplexityMetric(BaseMetric):
    def __init__(
        self,
        reference: str = "zero",
        eps: float = 1e-12,
        log_per_layer: bool = False,
    ):
        self.collector = BartlettCollector(reference=reference, eps=eps)
        self.log_per_layer = log_per_layer
        self.reset()

    def update(self, input, output, targets, model=None):
        self.model = _unwrap_model(model)

    def compute(self):
        if self.model is None:
            return {}

        model_metrics, per_layer = self.collector.collect_model(self.model)
        metrics = _numeric_items("bartlett", model_metrics)

        if self.log_per_layer:
            for name, layer_metrics in per_layer.items():
                metrics.update(
                    _numeric_items(
                        f"bartlett_layer_{_sanitize_layer_name(name)}",
                        layer_metrics,
                    )
                )

        return metrics

    def reset(self):
        self.model = None


# уверенность в действительно правильном результате(качество) / "сложность модели" - если у 2х моделей одинаковое качество,
# то модель с большей сложностью будет считаться лучшей из двух
class NormalizedMarginMetric(BaseMetric):
    def __init__(self, reference: str = "zero", eps: float = 1e-12):
        self.collector = BartlettCollector(reference=reference, eps=eps)
        self.eps = eps
        self.reset()

    def update(self, input, output, targets, model=None):
        self.model = _unwrap_model(model)

        if output is None or getattr(output, "logits", None) is None:
            return

        logits = output.logits.detach().float()
        targets = targets.detach()

        true_scores = logits.gather(1, targets[:, None]).squeeze(1)
        masked_logits = logits.clone()
        masked_logits[torch.arange(len(targets), device=targets.device), targets] = float("-inf")
        other_max = masked_logits.max(dim=1).values

        self.raw_margins.append((true_scores - other_max).cpu())
        self.x_sq_sum += input.detach().float().pow(2).sum().item()
        self.num_samples += input.shape[0]

    def compute(self):
        if self.model is None or not self.raw_margins or self.num_samples == 0:
            return {}

        bartlett_metrics, _ = self.collector.collect_model(self.model)
        spectral_complexity = max(bartlett_metrics.spectral_complexity, self.eps)
        denominator = max(spectral_complexity * ((self.x_sq_sum ** 0.5) / self.num_samples), self.eps)
        normalized_margins = torch.cat(self.raw_margins, dim=0) / denominator
        q25, q50, q75 = torch.quantile(
            normalized_margins,
            torch.tensor([0.25, 0.5, 0.75], dtype=normalized_margins.dtype),
        )

        return {
            "margin_mean": normalized_margins.mean().item(),
            "margin_std": normalized_margins.std(unbiased=False).item(),
            "margin_min": normalized_margins.min().item(),
            "margin_q25": q25.item(),
            "margin_median": q50.item(),
            "margin_q75": q75.item(),
            "margin_max": normalized_margins.max().item(),
            "margin_frac_positive": (normalized_margins > 0).float().mean().item(),
            "margin_denominator": float(denominator),
            "margin_spectral_complexity": float(spectral_complexity),
        }

    def reset(self):
        self.model = None
        self.raw_margins: list[torch.Tensor] = []
        self.x_sq_sum = 0.0
        self.num_samples = 0


class ComputeCostMetric(BaseMetric):
    def __init__(
        self,
        count_bias: bool = True,
        sample_size: int = 1,
        log_per_layer: bool = False,
    ):
        self.collector = ComputeCollector(count_bias=count_bias)
        self.sample_size = sample_size
        self.log_per_layer = log_per_layer
        self.reset()

    def update(self, input, output, targets, model=None):
        self.model = _unwrap_model(model)
        if self.input_sample is None:
            self.input_sample = input[: self.sample_size].detach().cpu()

    def compute(self):
        if self.model is None or self.input_sample is None:
            return {}

        model_metrics, per_layer = self.collector.collect_model(
            model=self.model,
            input_sample=self.input_sample,
        )
        metrics = _numeric_items("compute", model_metrics)

        if self.log_per_layer:
            for name, layer_metrics in per_layer.items():
                metrics.update(
                    _numeric_items(
                        f"compute_layer_{_sanitize_layer_name(name)}",
                        layer_metrics,
                    )
                )

        return metrics

    def reset(self):
        self.model = None
        self.input_sample = None
