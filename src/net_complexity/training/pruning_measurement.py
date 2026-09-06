"""Deployment measurements: explicit accounting scope, no test data."""
from __future__ import annotations

import hashlib
import json
import time

import torch
from torch import nn

from .interruption import check_stop


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def mask_hash(mask):
    return hashlib.sha256(json.dumps(mask, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.no_grad()
def evaluate_deployment(model, loader, device):
    model.to(device).eval()
    correct = count = 0
    ce_sum = 0.0
    for x, y in loader:
        check_stop()
        x, y = x.to(device), y.to(device)
        output = model(x, y)
        if not torch.isfinite(output.logits).all() or not torch.isfinite(output.ce_loss):
            raise FloatingPointError("Non-finite deployment output/CE.")
        correct += int((output.logits.argmax(-1) == y).sum())
        count += int(y.numel())
        ce_sum += float(output.ce_loss) * y.numel()
    if count == 0:
        raise ValueError("Empty validation set.")
    return {"accuracy": correct / count, "ce_loss": ce_sum / count,
            "correct_count": correct, "example_count": count}


@torch.no_grad()
def calibrate_bn(model, loader, device, batches, seed):
    if batches <= 0:
        return 0
    # Same train-only examples and augmentation RNG for old/candidate models.
    from .randomness import set_random_seed
    set_random_seed(seed)
    if loader.generator is not None:
        loader.generator.manual_seed(seed)
    model.to(device).train()
    bn_modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    momenta = [m.momentum for m in bn_modules]
    for module in bn_modules:
        module.reset_running_stats()
        module.momentum = None
    processed = 0
    try:
        for x, y in loader:
            check_stop()
            model(x.to(device), y.to(device))
            processed += 1
            if processed >= batches:
                break
    finally:
        for module, momentum in zip(bn_modules, momenta):
            module.momentum = momentum
        model.eval()
    if processed == 0:
        raise ValueError("Empty BN calibration loader.")
    return processed


@torch.no_grad()
def deployment_cost(model, *, image_shape=(3, 32, 32), device="cpu", latency=False):
    model.to(device).eval()
    backbone = getattr(model, "backbone", model)
    macs = 0
    handles = []

    def count(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        else:
            macs += output.numel() * module.in_features

    for module in backbone.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    try:
        backbone(torch.zeros((1, *image_shape), device=device))
    finally:
        for handle in handles:
            handle.remove()
    result = {
        "physical_total_parameters": sum(p.numel() for p in model.parameters()),
        "physical_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
        "conv_linear_macs_per_image": macs,
        "additional_dense_scatter_matmul_macs": 0,
        "mac_scope": "Conv/Linear only; scatter uses index_copy (not dense matmul)",
        "operations_not_counted_as_macs": ["BatchNorm", "ReLU", "pooling", "residual add", "index_copy"],
        "latency": None,
    }
    if latency and str(device).startswith("cuda"):
        measurements = {}
        for batch in (1, 128):
            check_stop()
            x = torch.zeros((batch, *image_shape), device=device)
            torch.cuda.reset_peak_memory_stats()
            for _ in range(10):
                backbone(x)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(30):
                backbone(x)
            torch.cuda.synchronize()
            measurements[str(batch)] = {
                "milliseconds_per_batch": (time.perf_counter() - start) * 1000 / 30,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        result["latency"] = measurements
    return result
