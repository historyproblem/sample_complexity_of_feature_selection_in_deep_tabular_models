from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class WeightMetrics:
    shape: tuple[int, ...]
    num_params: int
    l1_norm: float
    l2_norm: float
    fro_norm: float
    weight_sparsity: float


@dataclass
class BartlettLayerMetrics:
    shape: tuple[int, ...]
    spectral_norm: float
    ref_distance_21: float
    lipschitz_rho: float = 1.0


@dataclass
class BartlettModelMetrics:
    spectral_complexity: float
    lipschitz_product: float
    correction_term: float
    num_layers: int


@dataclass
class ComputeLayerMetrics:
    module_type: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    macs: int
    flops: int
    num_params: int


@dataclass
class ComputeModelMetrics:
    macs: int
    flops: int
    macs_per_sample: float
    flops_per_sample: float
    num_layers: int
    batch_size: int


class WeightCollector:
    def __init__(self, sparsity_eps: float = 1e-4):
        self.sparsity_eps = sparsity_eps

    def can_handle(self, module: nn.Module) -> bool:
        return hasattr(module, "weight") and module.weight is not None

    def collect(self, module: nn.Module) -> WeightMetrics:
        weights = module.weight.detach().float()

        return WeightMetrics(
            shape=tuple(weights.shape),
            num_params=weights.numel(),
            l1_norm=weights.abs().sum().item(),
            l2_norm=torch.linalg.vector_norm(weights.reshape(-1), ord=2).item(),
            fro_norm=torch.sqrt((weights ** 2).sum()).item(),
            weight_sparsity=(weights.abs() < self.sparsity_eps).float().mean().item(),
        )


class BartlettCollector:
    def __init__(self, reference: str = "zero", eps: float = 1e-12):
        self.reference = reference
        self.eps = eps

    def can_handle(self, module: nn.Module) -> bool:
        return (
            isinstance(module, (nn.Linear, nn.Conv2d))
            and hasattr(module, "weight")
            and module.weight is not None
        )

    def _as_matrix(self, weights: torch.Tensor) -> torch.Tensor:
        if weights.ndim == 2:
            return weights
        return weights.flatten(1)

    def _reference_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        if self.reference == "zero":
            return torch.zeros_like(matrix)

        if self.reference == "identity_if_square":
            if matrix.shape[0] == matrix.shape[1]:
                return torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
            return torch.zeros_like(matrix)

        raise ValueError(f"Unknown reference mode: {self.reference}")

    def _spectral_norm(self, matrix: torch.Tensor) -> float:
        return torch.linalg.matrix_norm(matrix, ord=2).item()

    def _norm_21_of_transpose_diff(self, matrix: torch.Tensor, reference: torch.Tensor) -> float:
        diff = matrix - reference
        return torch.linalg.vector_norm(diff, ord=2, dim=1).sum().item()

    def collect_layer(self, module: nn.Module) -> BartlettLayerMetrics:
        matrix = self._as_matrix(module.weight.detach().float())
        reference = self._reference_matrix(matrix)

        return BartlettLayerMetrics(
            shape=tuple(module.weight.shape),
            spectral_norm=self._spectral_norm(matrix),
            ref_distance_21=self._norm_21_of_transpose_diff(matrix, reference),
            lipschitz_rho=1.0,
        )

    def collect_model(
        self, model: nn.Module
    ) -> tuple[BartlettModelMetrics, dict[str, BartlettLayerMetrics]]:
        per_layer: dict[str, BartlettLayerMetrics] = {}

        for name, module in model.named_modules():
            if self.can_handle(module):
                per_layer[name] = self.collect_layer(module)

        if not per_layer:
            raise ValueError("No supported layers found (Linear/Conv2d).")

        lipschitz_product = 1.0
        correction_sum = 0.0

        for layer_metrics in per_layer.values():
            spectral_norm = max(layer_metrics.spectral_norm, self.eps)
            lipschitz_product *= layer_metrics.lipschitz_rho * spectral_norm
            correction_sum += (layer_metrics.ref_distance_21 / spectral_norm) ** (2.0 / 3.0)

        correction_term = correction_sum ** (3.0 / 2.0)

        return BartlettModelMetrics(
            spectral_complexity=float(lipschitz_product * correction_term),
            lipschitz_product=float(lipschitz_product),
            correction_term=float(correction_term),
            num_layers=len(per_layer),
        ), per_layer


class ComputeCollector:
    def __init__(self, count_bias: bool = True):
        self.count_bias = count_bias

    def can_handle(self, module: nn.Module) -> bool:
        return isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))

    def _first_tensor(self, value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value

        if isinstance(value, (tuple, list)):
            for item in value:
                if torch.is_tensor(item):
                    return item

        raise TypeError("No tensor found in hook inputs/outputs.")

    def _shape(self, value: Any) -> tuple[int, ...]:
        return tuple(self._first_tensor(value).shape)

    def _num_params(self, module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters(recurse=False))

    def _count_linear(self, module: nn.Linear, output: Any) -> tuple[int, int]:
        out = self._first_tensor(output)
        num_outputs = out.numel()

        macs = num_outputs * module.in_features
        flops = 2 * macs

        if self.count_bias and module.bias is not None:
            flops += num_outputs

        return int(macs), int(flops)

    def _count_conv(self, module: nn.Module, output: Any) -> tuple[int, int]:
        out = self._first_tensor(output)
        num_outputs = out.numel()
        kernel_ops = module.weight[0].numel()

        macs = num_outputs * kernel_ops
        flops = 2 * macs

        if self.count_bias and module.bias is not None:
            flops += num_outputs

        return int(macs), int(flops)

    def _count_module(self, module: nn.Module, output: Any) -> tuple[int, int]:
        if isinstance(module, nn.Linear):
            return self._count_linear(module, output)

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return self._count_conv(module, output)

        return 0, 0

    @torch.no_grad()
    def collect_model(
        self,
        model: nn.Module,
        input_sample: torch.Tensor | tuple[Any, ...] | list[Any],
        device: torch.device | None = None,
    ) -> tuple[ComputeModelMetrics, dict[str, ComputeLayerMetrics]]:
        was_training = model.training
        model.eval()

        if device is None:
            device = next(model.parameters()).device

        if torch.is_tensor(input_sample):
            model_args = (input_sample.to(device),)
        elif isinstance(input_sample, (tuple, list)):
            model_args = tuple(
                item.to(device) if torch.is_tensor(item) else item for item in input_sample
            )
        else:
            raise TypeError("input_sample must be a Tensor or tuple/list of args.")

        per_layer: dict[str, ComputeLayerMetrics] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inputs: Any, output: Any):
                macs, flops = self._count_module(module, output)
                current = ComputeLayerMetrics(
                    module_type=module.__class__.__name__,
                    input_shape=self._shape(inputs),
                    output_shape=self._shape(output),
                    macs=macs,
                    flops=flops,
                    num_params=self._num_params(module),
                )

                if name in per_layer:
                    per_layer[name].macs += current.macs
                    per_layer[name].flops += current.flops
                else:
                    per_layer[name] = current

            return hook

        for name, module in model.named_modules():
            if self.can_handle(module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        try:
            model(*model_args)
        finally:
            for hook in hooks:
                hook.remove()
            if was_training:
                model.train()

        if not per_layer:
            raise ValueError("No supported layers found for compute counting.")

        total_macs = sum(layer_metrics.macs for layer_metrics in per_layer.values())
        total_flops = sum(layer_metrics.flops for layer_metrics in per_layer.values())
        batch_size = self._first_tensor(model_args).shape[0]

        return ComputeModelMetrics(
            macs=int(total_macs),
            flops=int(total_flops),
            macs_per_sample=float(total_macs / batch_size),
            flops_per_sample=float(total_flops / batch_size),
            num_layers=len(per_layer),
            batch_size=int(batch_size),
        ), per_layer
