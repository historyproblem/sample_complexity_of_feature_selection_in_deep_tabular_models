from __future__ import annotations

import hashlib
import statistics
import time

import torch
import torch.nn.functional as F

from net_complexity.metrics.stochastic_depth import StochasticDepthFLOPsMetric
from net_complexity.models.stochastic_depth import (
    HuangStochasticDepthInvertedResidual,
    get_stochastic_depth_blocks,
)


def state_digest(model: torch.nn.Module) -> str:
    """Hash parameters and persistent buffers, including BatchNorm statistics."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def evaluate_stochastic_depth(
    model: torch.nn.Module,
    dataloader,
    *,
    device: str = "cpu",
    mode: str = "stochastic",
    seed: int = 42,
) -> dict:
    """One prediction per image; sample branches per batch, with BN/dropout frozen.

    Repeats belong outside this function. Predictions from multiple masks are
    never averaged into an ensemble. Active parameters count executed branches,
    not the size of the checkpoint, which always stores the complete model.
    """
    if mode not in {"expected", "stochastic"}:
        raise ValueError("mode must be 'expected' or 'stochastic'.")
    blocks = get_stochastic_depth_blocks(model)
    if not blocks or any(
        not isinstance(block, HuangStochasticDepthInvertedResidual)
        for block in blocks.values()
    ):
        raise ValueError("Expected MobileNetV2 stochastic-depth identity branches.")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and device_obj.index is None:
        device_obj = torch.device("cuda", torch.cuda.current_device())
    if any(parameter.device != device_obj for parameter in model.parameters()):
        raise ValueError("Move the model to the requested device before evaluation.")
    previous_training = {module: module.training for module in model.modules()}
    previous_modes = {name: block.eval_mode for name, block in blocks.items()}
    before = state_digest(model)
    full_params = sum(parameter.numel() for parameter in model.parameters())
    branch_params = {
        name: sum(parameter.numel() for parameter in block.branch.parameters())
        for name, block in blocks.items()
    }
    always_params = full_params - sum(branch_params.values())
    executed_samples = {name: 0 for name in blocks}
    correct = examples = batches = 0
    ce_sum = 0.0
    flops = StochasticDepthFLOPsMetric(sample_size=1)
    cuda_devices = (
        list(range(torch.cuda.device_count())) if device_obj.type == "cuda" else []
    )
    try:
        model.eval()
        for block in blocks.values():
            block.eval_mode = mode
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            started = time.perf_counter()
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device_obj), labels.to(device_obj)
                logits = model(inputs)
                if not torch.is_tensor(logits) or not torch.isfinite(logits).all():
                    raise ValueError("Expected finite classifier logits.")
                count = int(labels.numel())
                correct += int((logits.argmax(dim=-1) == labels).sum().item())
                ce_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
                examples += count
                batches += 1
                for name, block in blocks.items():
                    executed_samples[name] += (
                        int(block.last_residual_branch_active) * count
                    )
                flops.update(inputs, None, labels, model)
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            elapsed = time.perf_counter() - started
            if not examples:
                raise ValueError("Cannot evaluate an empty dataset.")
            compute = flops.compute()
        after = state_digest(model)
        if after != before:
            raise RuntimeError(
                "Frozen evaluation changed parameters or persistent buffers."
            )
        expected_probabilities = {
            name: (block.survival_probability if mode == "stochastic" else 1.0)
            for name, block in blocks.items()
        }
        return {
            "mode": mode,
            "seed": int(seed),
            "accuracy": correct / examples,
            "correct_count": correct,
            "example_count": examples,
            "ce_loss": ce_sum / examples,
            "batches": batches,
            "elapsed_seconds": elapsed,
            "stored_parameters": full_params,
            "mean_active_parameters": always_params
            + sum(
                branch_params[name] * executed_samples[name] / examples
                for name in blocks
            ),
            "expected_active_parameters": always_params
            + sum(
                branch_params[name] * expected_probabilities[name] for name in blocks
            ),
            "mean_active_blocks": sum(executed_samples.values()) / examples,
            "expected_active_blocks": sum(expected_probabilities.values()),
            "mean_forward_flops_per_image": compute[
                "stochastic_depth_actual_inference_forward_flops_per_sample"
            ],
            "expected_forward_flops_per_image": compute[
                "stochastic_depth_expected_inference_forward_flops_per_sample"
            ],
            "full_forward_flops_per_image": compute[
                "stochastic_depth_full_inference_flops_per_sample"
            ],
            "branch_sample_execution_counts": executed_samples,
            "model_state_sha256": before,
            "parameters_and_bn_unchanged": True,
        }
    finally:
        for name, block in blocks.items():
            block.eval_mode = previous_modes[name]
        for module, training in previous_training.items():
            module.training = training


def summarize_stochastic_repeats(repeats: list[dict]) -> dict:
    if not repeats or any(row["mode"] != "stochastic" for row in repeats):
        raise ValueError("Expected one or more stochastic inference repeats.")
    result = {"repeats": len(repeats), "seeds": [row["seed"] for row in repeats]}
    for key in (
        "accuracy",
        "ce_loss",
        "mean_active_parameters",
        "mean_active_blocks",
        "mean_forward_flops_per_image",
    ):
        values = [row[key] for row in repeats]
        result[key + "_mean"] = statistics.mean(values)
        result[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result
