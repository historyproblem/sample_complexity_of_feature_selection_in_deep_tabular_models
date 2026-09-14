"""Deployment measurements: explicit accounting scope, no test data."""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from copy import deepcopy
import random

import numpy as np

import torch
from torch import nn

from .interruption import check_stop


@contextmanager
def isolated_diagnostic_rng(*loaders):
    """Keep diagnostic iteration and calibration outside the training RNG stream.

    Callers must supply separate loaders for worker-based augmentation: worker
    process RNG cannot be rewound by a parent-process snapshot.
    """
    py, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generators = [(loader.generator, loader.generator.get_state()) for loader in loaders
                  if getattr(loader, "generator", None) is not None]
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for generator, state in generators:
            generator.set_state(state)


@torch.no_grad()
def compare_predictors(physical, carrier, loader, device="cpu"):
    """Measure surgery and gate-removal effects on exactly the same examples.

    Copies ensure original mode, runtime and BN buffers are unchanged. No BN
    calibration is performed here. The all-open equivalence check is distinct
    from the carried-gates predictor comparison.
    """
    from net_complexity.models.pruning_budget import gates
    start = time.perf_counter()
    before = [state_hash(m.state_dict()) for m in (physical, carrier)]
    with isolated_diagnostic_rng(loader):
        models = [deepcopy(physical).to(device).eval(), deepcopy(carrier).to(device).eval(),
                  deepcopy(carrier).to(device).eval()]
        for gate in gates(models[1]).values():
            gate.set_bypass(True)
        for gate in gates(models[2]).values():
            gate.get_hard_gate_decisions()  # reject an incompatible predictor; preserve exact restored runtime
        counts = [dict(correct=0, ce_sum=0.0) for _ in models]
        comparisons = [dict(max_abs=0.0, squared_error=0.0, logits=0, disagreement=0)
                       for _ in range(2)]
        examples = batches = 0
        for x, y in loader:
            check_stop()
            x, y = x.to(device), y.to(device)
            outputs = [model(x, y) for model in models]
            if any(not torch.isfinite(out.logits).all() or not torch.isfinite(out.ce_loss)
                   for out in outputs):
                raise FloatingPointError("Non-finite predictor diagnostic.")
            examples += y.numel()
            batches += 1
            for out, count in zip(outputs, counts):
                count["correct"] += int((out.logits.argmax(-1) == y).sum())
                count["ce_sum"] += float(out.ce_loss) * y.numel()
            for out, stats in zip(outputs[1:], comparisons):
                delta = out.logits.double() - outputs[0].logits.double()
                stats["max_abs"] = max(stats["max_abs"], float(delta.abs().max()))
                stats["squared_error"] += float(delta.square().sum())
                stats["logits"] += delta.numel()
                stats["disagreement"] += int((out.logits.argmax(-1) != outputs[0].logits.argmax(-1)).sum())
        if examples == 0:
            raise ValueError("Empty diagnostic validation loader.")
        result = {name: {"accuracy": c["correct"] / examples, "ce_loss": c["ce_sum"] / examples}
                  for name, c in zip(("physical", "all_open_carrier", "carry_carrier"), counts)}
        for name, stats in zip(("all_open_vs_physical", "carry_vs_physical"), comparisons):
            result[name] = {"logits_max_abs": stats["max_abs"],
                            "logits_rmse": (stats["squared_error"] / stats["logits"]) ** .5,
                            "prediction_disagreement": stats["disagreement"] / examples}
    if before != [state_hash(m.state_dict()) for m in (physical, carrier)]:
        raise AssertionError("Diagnostic changed predictor state.")
    result["overhead"] = {"examples_per_predictor": examples, "forward_examples": 3 * examples,
                           "batches_per_predictor": batches, "wall_seconds": time.perf_counter() - start,
                           "bn_calibration": False, "training_rng_preserved": True}
    return result


@torch.no_grad()
def gated_export_equivalence(carrier, physical, sample, device="cpu"):
    """Check the actual gated predictor separately, before BN calibration.

    Preserve the audited bounded FP32/FP64 diagnostic tolerance policy. Double
    precision runs only on disposable copies and does not change training dtype.
    """
    from .pruning_audit import _equivalence_stats
    started = time.perf_counter()
    with isolated_diagnostic_rng():
        gated = deepcopy(carrier).to(device).eval()
        exported = deepcopy(physical).to(device).eval()
        x, y = (tensor.to(device) for tensor in sample)
        actual, expected = gated(x, y).logits, exported(x, y).logits
        report = {"policy": "fp32_then_bounded_fp64_v1", "predictor": "selected_gated_runtime",
                  "fp32": _equivalence_stats(actual, expected, rtol=1e-4, atol=1e-5)}
        if report["fp32"]["mismatched_logits"] == 0:
            report["status"] = "passed_fp32"
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            report["fp32_ceiling"] = {"rtol": 1e-4, "atol": 1e-4}
            gated, exported = gated.cpu().double(), exported.cpu().double()
            x64, y64 = sample[0].cpu().double(), sample[1].cpu()
            actual64 = torch.cat([gated(x64[i:i+8], y64[i:i+8]).logits for i in range(0, len(x64), 8)])
            expected64 = torch.cat([exported(x64[i:i+8], y64[i:i+8]).logits for i in range(0, len(x64), 8)])
            report["fp64"] = _equivalence_stats(actual64, expected64, rtol=1e-8, atol=1e-9)
            torch.testing.assert_close(actual64, expected64, rtol=1e-8, atol=1e-9)
            report["status"] = "passed_fp64_fallback"
    report["wall_seconds"] = time.perf_counter() - started
    report["forward_examples"] = len(sample[1]) * (4 if "fp64" in report else 2)
    return report


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
def calibrate_bn(model, loader, device, batches, seed, *, accounting=None):
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
    examples = 0
    started = time.perf_counter()
    try:
        for x, y in loader:
            check_stop()
            model(x.to(device), y.to(device))
            processed += 1
            examples += int(y.numel())
            if processed >= batches:
                break
    finally:
        for module, momentum in zip(bn_modules, momenta):
            module.momentum = momentum
        model.eval()
        if accounting is not None:
            accounting.update(batches=processed, forward_examples=examples,
                              wall_seconds=time.perf_counter() - started)
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
