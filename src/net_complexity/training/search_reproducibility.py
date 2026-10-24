"""Read-only observations of the actual search batches and numerical trajectory.

No probes, random draws, model forwards, backend changes or BatchNorm updates are
performed here. CPU copies and synchronization add diagnostic overhead only.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess

import numpy as np
import torch
from torch import nn


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _json_hash(value):
    payload = json.dumps(_json_value(value), sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_record(value, *, numeric=False):
    tensor = value.detach().cpu().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    result = {"sha256": hashlib.sha256(raw).hexdigest(),
              "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    if numeric and tensor.numel():
        values = tensor.to(torch.float64)
        finite = torch.isfinite(values)
        result["finite_count"] = int(finite.sum())
        if bool(finite.all()):
            result.update(min=float(values.min()), max=float(values.max()),
                          mean=float(values.mean()), l2=float(torch.linalg.vector_norm(values)))
    return result


def _named_records(named_tensors, *, numeric=False):
    records = {name: tensor_record(value, numeric=numeric)
               for name, value in named_tensors if value is not None}
    identity = {name: {key: value for key, value in record.items()
                       if key in {"sha256", "dtype", "shape"}}
                for name, record in records.items()}
    return _json_hash(identity), records


def _bn_tensors(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for key, value in module.named_buffers(recurse=False):
                if value is not None:
                    yield f"{name}.{key}", value


def _rng_record(loader):
    numpy_state = np.random.get_state()
    result = {
        "python": _json_hash(random.getstate()),
        "numpy": _json_hash(numpy_state),
        "torch_cpu": tensor_record(torch.get_rng_state())["sha256"],
        "cuda": [tensor_record(state)["sha256"]
                 for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [],
    }
    generator = getattr(loader, "generator", None)
    if generator is not None:
        result["train_dataloader"] = tensor_record(generator.get_state())["sha256"]
    return result


def runtime_environment():
    """Record effective policy; never silently enable deterministic execution."""
    try:
        torchvision_version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError:
        torchvision_version = None
    devices = []
    driver_metadata = {"status": "not_available", "devices": []}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name,
                            "capability": [props.major, props.minor],
                            "total_memory": props.total_memory,
                            "multi_processor_count": props.multi_processor_count})
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            try:
                query = subprocess.run(
                    [nvidia_smi, "--query-gpu=index,uuid,driver_version", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                rows = [line.split(",") for line in query.stdout.splitlines() if line.strip()]
                if query.returncode == 0 and all(len(row) == 3 for row in rows):
                    driver_metadata = {"status": "measured", "devices": [
                        dict(zip(("physical_index", "uuid", "driver_version"),
                                 (value.strip() for value in row))) for row in rows
                    ]}
            except (OSError, subprocess.TimeoutExpired):
                driver_metadata["status"] = "query_unavailable"
    return {
        "schema_version": 1, "python_version": platform.python_version(),
        "platform": platform.platform(), "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version, "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "cuda_devices": devices, "num_threads": torch.get_num_threads(),
        "nvidia_driver": driver_metadata,
        "num_interop_threads": torch.get_num_interop_threads(),
        "backend_policy": {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
        },
        "environment_overrides": {key: os.environ.get(key) for key in (
            "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "CUDA_VISIBLE_DEVICES",
        )},
        "backend_policy_changed_by_observer": False,
    }


class SearchReproducibilityTrace:
    def __init__(self, output_dir, *, rich_batches=3, rich_epochs=(1, 11)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "trace.jsonl"
        if self.path.exists() or (self.output_dir / "environment.json").exists():
            raise FileExistsError("Diagnostic output already exists; use a fresh trace directory")
        if type(rich_batches) is not int or rich_batches < 0:
            raise ValueError("rich_batches must be a nonnegative integer")
        self.rich_batches = rich_batches
        self.rich_epochs = frozenset(int(epoch) for epoch in rich_epochs)
        environment = runtime_environment()
        environment["trace_policy"] = {
            "all_training_batch_inputs": True, "rng_first_batch_each_epoch": True,
            "rich_batches_per_epoch": rich_batches, "rich_epochs": sorted(self.rich_epochs),
        }
        (self.output_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        self.path.touch(exist_ok=False)

    def _write(self, kind, epoch, *, batch_index=None, **payload):
        record = {"kind": kind, "epoch": int(epoch), **payload}
        if batch_index is not None:
            record["batch_index"] = int(batch_index)
        with self.path.open("a") as stream:
            stream.write(json.dumps(_json_value(record), sort_keys=True, allow_nan=False) + "\n")

    def _rich(self, epoch, batch_index):
        return epoch in self.rich_epochs and batch_index < self.rich_batches

    def on_batch_input(self, epoch, batch_index, inputs, labels, loader):
        extra = {}
        if batch_index == 0:
            extra["rng"] = _rng_record(loader)
            extra["loader"] = {
                "num_workers": loader.num_workers,
                "persistent_workers": getattr(loader, "persistent_workers", False),
                "batch_size": loader.batch_size, "drop_last": loader.drop_last,
            }
        self._write("batch_input", epoch, batch_index=batch_index,
                    inputs=tensor_record(inputs, numeric=self._rich(epoch, batch_index)),
                    labels=tensor_record(labels, numeric=self._rich(epoch, batch_index)), **extra)

    def on_forward(self, epoch, batch_index, output):
        if self._rich(epoch, batch_index):
            fields = {name: tensor_record(value, numeric=True)
                      for name in ("logits", "loss", "ce_loss", "reg_loss")
                      if isinstance(value := getattr(output, name, None), torch.Tensor)}
            self._write("forward", epoch, batch_index=batch_index, outputs=fields)

    def on_backward(self, epoch, batch_index, model):
        if self._rich(epoch, batch_index):
            digest, records = _named_records(
                ((name, parameter.grad) for name, parameter in model.named_parameters()),
                numeric=True,
            )
            self._write("backward", epoch, batch_index=batch_index,
                        gradients_sha256=digest, gradients=records)

    def on_optimizer_step(self, epoch, batch_index, model):
        if self._rich(epoch, batch_index):
            digest, records = _named_records(model.state_dict().items(), numeric=True)
            self._write("optimizer_step", epoch, batch_index=batch_index,
                        model_state_sha256=digest, model_state=records)

    def on_epoch_end(self, epoch, model, train_metrics, valid_metrics, controller, ledger):
        model_digest, _ = _named_records(model.state_dict().items())
        trainable_digest, _ = _named_records(
            (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        bn_digest, bn_records = _named_records(_bn_tensors(model), numeric=True)
        self._write("epoch_end", epoch, model_state_sha256=model_digest,
                    trainable_state_sha256=trainable_digest, bn_state_sha256=bn_digest,
                    bn_state=bn_records, train_metrics=dict(train_metrics),
                    valid_metrics=dict(valid_metrics),
                    controller_state=controller.state_dict() if controller is not None else None,
                    ledger=dict(ledger))
