"""One learned-mask extraction, then genuinely fresh compact-model training.

This protocol deliberately has a 150 + 150 epoch cost, not the cyclic pilot's
150-epoch total. Only topology crosses the phase boundary. The dense reference
is an explicit, separately charged, continuous 150-epoch experiment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from net_complexity.models.channel_pruning import (
    build_structurally_pruned_model_from_config, transfer_gated_weights_to_structural,
)
from net_complexity.models.feature_selection import GumbelLayer, MaskedGumbelLayer
from net_complexity.models.pruning_budget import gates, validate_mask
from .cyclic_aig import _configure_run_history, _set_num_epochs
from .engine import run_training
from .interruption import TrainingInterrupted, check_stop, cooperative_signals
from .pruning_audit import committed_equivalence, load_adaptive_reference
from .pruning_measurement import deployment_cost, evaluate_deployment, mask_hash, state_hash, write_json
from .randomness import set_random_seed

PROTOCOL = "one_shot_reinit_v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_config(config, *, allow_short_run=False):
    """Fail closed on cyclic, budgeted, pretrained, or test-selected variants."""
    c = config.one_shot_pruning
    allowed = {"protocol", "search_epochs", "retrain_epochs", "mask_threshold",
               "selection_checkpoint", "probability_source", "reinit_seed",
               "initial_checkpoint", "reference_history"}
    if set(c) != allowed:
        raise ValueError(f"one_shot_pruning keys must be exactly {sorted(allowed)}")
    for key, expected in {"protocol": PROTOCOL, "selection_checkpoint": "best.pt",
                          "probability_source": "raw_logits", "mask_threshold": 0.5}.items():
        if c[key] != expected:
            raise ValueError(f"one_shot_pruning.{key} must be {expected!r}")
    for key in ("search_epochs", "retrain_epochs"):
        if type(c[key]) is not int or c[key] < 1 or (not allow_short_run and c[key] != 150):
            raise ValueError(f"{key} must be 150 (short smoke requires explicit allow_short_run=True)")
    if type(c.reinit_seed) is not int or c.reinit_seed < 0:
        raise ValueError("reinit_seed must be a nonnegative integer")
    required = {
        "training_arguments.evaluate_test": False,
        "dataloaders.include_test": False,
        "training_arguments.adaptive_lambda.enabled": True,
        "training_arguments.lambda_warmup.enabled": False,
        "training_arguments.batchnorm_recalibration.enabled": False,
        "model.entropy_regularization_coef": 0.0,
        "optimizer.gate_weight_decay_scale": 0.0,
        "model.backbone.resnet_block.train_gate_mode": "ste_hard",
        "model.backbone.resnet_block.eval_gate_mode": "deterministic_hard",
        "model.backbone.resnet_block.gate_threshold": 0.5,
        "model.backbone.resnet_block.regularization_normalization": "initial_channels",
        "run_history.monitor": "valid_accuracy",
        "run_history.secondary_monitor": "valid_ce_loss",
        "scheduler._target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
    }
    for key, expected in required.items():
        if OmegaConf.select(config, key) != expected:
            raise ValueError(f"One-shot requires {key}={expected!r}")
    for key in ("cyclic_channel_pruning", "cyclic_layer_dropping", "depgraph_pruning",
                "aig_static_pruning", "channel_pruning", "layer_skipping"):
        if OmegaConf.select(config, key + ".enabled", default=False):
            raise ValueError(f"One-shot cannot enable legacy {key}")
    for key in ("early_stopping", "collapse_guard", "gate_mode_schedule"):
        if OmegaConf.select(config, "training_arguments." + key + ".enabled", default=False):
            raise ValueError(f"One-shot does not support {key}")
    if OmegaConf.select(config, "training_arguments.adaptive_lambda.baseline_history_dir") not in (None, ""):
        raise ValueError("Explicit dense reference required; hidden baseline training is forbidden")
    if not math.isfinite(float(config.model.lambda_coef)) or config.model.lambda_coef <= 0:
        raise ValueError("Search lambda must be positive and finite")
    return int(c.search_epochs + c.retrain_epochs)


def _training_signature(config):
    """Settings that a reusable dense curve must share with each search job."""
    cfg = OmegaConf.to_container(config, resolve=True)
    backbone = deepcopy(cfg["model"]["backbone"])
    backbone.pop("resnet_block", None)  # Dense has no selector; internal/output may differ.
    data = deepcopy(cfg["dataloaders"])
    for key in ("path_to_data", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor"):
        data.pop(key, None)
    scheduler = deepcopy(cfg["scheduler"])
    scheduler["T_max"] = int(config.one_shot_pruning.search_epochs)
    signature = {"backbone": backbone, "criterion": cfg["model"].get("criterion"),
                 "optimizer": cfg["optimizer"], "scheduler": scheduler, "dataloaders": data,
                 "seed": int(config.seed), "epochs": int(config.one_shot_pruning.search_epochs)}
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def validate_dense_reference(source, config, *, allow_short_run=False):
    """Verify a separately completed *continuous* reference, not old staged J1."""
    validate_config(config, allow_short_run=allow_short_run)
    source = Path(source).resolve()
    state = json.loads((source / "one_shot_state.json").read_text())
    epochs = int(config.one_shot_pruning.search_epochs)
    expected = {"protocol": PROTOCOL, "kind": "dense_reference", "status": "completed",
                "dense_epochs_completed": epochs, "global_epochs_completed": epochs,
                "total_epochs_allocated": epochs, "seed": int(config.seed),
                "search_epochs_completed": 0, "retrain_epochs_completed": 0,
                "training_signature": _training_signature(config), "schedule": "single_continuous_cosine"}
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Dense reference mismatch: {key}")
    resolved = source / "resolved_config.yaml"
    if state.get("resolved_config_sha256") != _sha(resolved):
        raise ValueError("Dense resolved configuration file changed")
    saved_config = OmegaConf.load(resolved)
    validate_config(saved_config, allow_short_run=allow_short_run)
    if _training_signature(saved_config) != state["training_signature"]:
        raise ValueError("Dense configuration/signature mismatch")
    stages = state.get("stages", [])
    if len(stages) != 1 or any(stages[0].get(key) != expected for key, expected in {
        "name": "dense", "status": "completed", "epochs_allocated": epochs,
        "epochs_completed": epochs, "scheduler": "single_continuous_cosine", "scheduler_T_max": epochs,
        "optimizer_state_reused": False, "adaptive_controller_reused": False, "regularization_active": False,
    }.items()):
        raise ValueError("Dense reference must be one completed, continuous, ungated phase")
    initializer, history = source / "initializer.pt", source / "global_history.csv"
    if state.get("initial_checkpoint_sha256") != _sha(initializer):
        raise ValueError("Dense zero-epoch initializer file changed")
    if state.get("reference_history_sha256") != _sha(history):
        raise ValueError("Dense validation history changed")
    initial = _load_initial(initializer, int(config.seed))
    if initial["model_state_hash"] != state.get("common_init_hash"):
        raise ValueError("Dense common initialization hash mismatch")
    load_adaptive_reference(history, epochs)
    steps_total = state.get("optimizer_steps_total", 0)
    if type(steps_total) is not int or steps_total <= 0 or steps_total % epochs:
        raise ValueError("Dense optimizer step accounting is invalid")
    with history.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != epochs:
        raise ValueError("Dense history must contain exactly the allocated epochs")
    for epoch, row in enumerate(rows, start=1):
        if (int(row["global_epoch"]) != epoch or int(row["local_epoch"]) != epoch
                or row["stage"] != "dense" or int(row["data_epoch"]) != epoch
                or int(row["data_seed"]) != int(config.seed)
                or int(row["optimizer_steps_total"]) != epoch * (steps_total // epochs)
                or float(row["lambda_used"]) != 0 or float(row["lambda_next"]) != 0
                or not math.isfinite(float(row["valid_ce_loss"])) or float(row["valid_ce_loss"]) < 0):
            raise ValueError(f"Dense history is not continuous ungated training at epoch {epoch}")
    checkpoint = source / "deployment.pt"
    if state.get("deployment_sha256") != _sha(checkpoint):
        raise ValueError("Dense deployment file changed")
    deployment = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (deployment.get("protocol") != PROTOCOL or deployment.get("kind") != "dense_reference"
            or deployment.get("global_epochs_consumed") != epochs
            or deployment.get("pruning_mask") != {} or state.get("accepted_mask") != {}
            or deployment.get("mask_hash") != mask_hash({}) or state.get("accepted_mask_hash") != mask_hash({})
            or deployment.get("common_init_hash") != state["common_init_hash"]
            or deployment.get("validation") != state.get("validation")):
        raise ValueError("Dense deployment/manifest metadata mismatch")
    deployment_hash = state_hash(deployment["model_state_dict"])
    if deployment_hash != state.get("deployment_model_state_hash") or deployment_hash != deployment.get("model_state_hash"):
        raise ValueError("Dense deployment state hash mismatch")
    _check_finite_state(deployment["model_state_dict"])
    verification_model = _fresh_compact(config, {}, int(config.seed))
    verification_model.load_state_dict(deployment["model_state_dict"], strict=True)
    count = sum(parameter.numel() for parameter in verification_model.parameters())
    if (state.get("final_cost", {}).get("physical_total_parameters") != count
            or state.get("initial_cost", {}).get("physical_total_parameters") != count):
        raise ValueError("Dense physical parameter count mismatch")
    state.update(initial_checkpoint=str(initializer), reference_history=str(history))
    return state


def _load_initial(path, seed):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("trained_epochs") != 0 or payload.get("seed") != seed:
        raise ValueError("Initializer must have zero trained epochs and the matching search seed")
    actual = state_hash(payload["model_state_dict"])
    if payload.get("model_state_hash") != actual:
        raise ValueError("Initializer state hash mismatch")
    _check_finite_state(payload["model_state_dict"])
    for key, tensor in payload["model_state_dict"].items():
        if "gumbel_layer" in key:
            raise ValueError("Shared initializer must be a plain, ungated backbone")
        if key.endswith(("running_mean", "num_batches_tracked")) and torch.count_nonzero(tensor):
            raise ValueError("Zero-epoch initializer contains trained BatchNorm statistics")
        if key.endswith("running_var") and not torch.equal(tensor, torch.ones_like(tensor)):
            raise ValueError("Zero-epoch initializer contains trained BatchNorm variance")
    return payload


def _check_finite_state(state):
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in state.values()):
        raise FloatingPointError("Non-finite model/checkpoint state")


def _load_plain_initial(model, state):
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(
        "gumbel_layer" not in key and not key.endswith("active_indices")
        for key in incompatible.missing_keys
    ):
        raise ValueError(f"Incompatible zero-epoch backbone initializer: {incompatible}")


def mask_from_raw_logits(model, threshold=0.5):
    """Inclusive p_open <= threshold; no floor rescue, ranking, or budget.

    Recovery's temporary open_bias is intentionally excluded. Diagnostics list
    every boundary before rejecting collapsed topology, rather than silently
    reopening a channel that the learned mask closed.
    """
    if not math.isfinite(float(threshold)) or not 0 <= threshold <= 1:
        raise ValueError("Invalid mask threshold")
    selectors = gates(model)
    if not selectors:
        raise ValueError("Search model has no masked channel gates")
    mask, boundaries, invalid = {}, {}, []
    for name, gate in selectors.items():
        if not torch.equal(gate.channel_mask, torch.ones_like(gate.channel_mask)):
            raise ValueError(f"Search must start without permanent pruning: {name}")
        logits = gate.logits.detach().float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite learned logits at {name}")
        probabilities = logits.softmax(-1)[:, 1].cpu()
        removed = (probabilities <= threshold).nonzero(as_tuple=False).flatten().tolist()
        if removed:
            mask[name] = removed
        remaining = len(probabilities) - len(removed)
        boundaries[name] = {"original_channels": len(probabilities), "removed_channels": len(removed),
                            "remaining_channels": remaining, "p_open": probabilities.tolist()}
        if remaining == 0:
            invalid.append(name)
    report = {"probability_source": "raw_logits", "temporary_open_bias_excluded": True,
              "comparison": "p_open <= threshold", "threshold": float(threshold),
              "parameter_budget": None, "minimum_keep_ratio": None,
              "collapsed_boundaries": invalid, "boundaries": boundaries,
              "removed_gate_channels": sum(map(len, mask.values())), "mask": mask,
              "mask_hash": mask_hash(mask), "status": "collapsed" if invalid else "selected"}
    return mask, report


def _fresh_compact(config, mask, seed):
    """Build from constructors only: no source model or checkpoint argument."""
    set_random_seed(seed)
    cfg = deepcopy(config)
    cfg.model.lambda_coef = 0.0
    pruning = OmegaConf.create({"enabled": True, "mode": "explicit", "structural": True, "mask": mask})
    model = build_structurally_pruned_model_from_config(cfg, pruning)
    assert_fresh_compact(model)
    return model


def assert_fresh_compact(model):
    if any(isinstance(module, (GumbelLayer, MaskedGumbelLayer)) for module in model.modules()):
        raise ValueError("Scratch model must contain no Gumbel gates")
    if float(model.lambda_coef) != 0:
        raise ValueError("Scratch model must have zero gate penalty")
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.running_mean is not None and (torch.count_nonzero(module.running_mean)
                    or not torch.equal(module.running_var, torch.ones_like(module.running_var))
                    or int(module.num_batches_tracked) != 0):
                raise ValueError("Scratch BatchNorm must have fresh statistics")
            if module.affine and (not torch.equal(module.weight, torch.ones_like(module.weight))
                                  or torch.count_nonzero(module.bias)):
                raise ValueError("Scratch BatchNorm affine parameters must be fresh")
    _check_finite_state(model.state_dict())


def _data(config):
    data = instantiate(config.dataloaders, include_test=False, loader_seed=int(config.seed))
    if str(config.dataloaders._target_).endswith("ClassicCVDataloaders"):
        if (len(data.train_dataloader.dataset), len(data.valid_dataloader.dataset)) != (45000, 5000):
            raise ValueError("CIFAR10 requires matched 45000/5000 train/validation splits")
        if str(config.device).startswith("cuda") and data.train_dataloader.num_workers == 0:
            raise ValueError("CUDA search requires worker-isolated augmentation RNG")
    split = mask_hash({name: list(getattr(loader.dataset, "indices", range(len(loader.dataset))))
                      for name, loader in (("train", data.train_dataloader), ("valid", data.valid_dataloader))})
    return data, split


def _phase_config(config, root, name, epochs, mask):
    cfg = deepcopy(config)
    _set_num_epochs(cfg, epochs)
    # The scratch phase resets BOTH model RNG and epoch-based data ordering.
    phase_seed = int(config.one_shot_pruning.reinit_seed) if name == "scratch" else int(config.seed)
    cfg.seed = phase_seed
    cfg.training_arguments.global_epoch_offset = 0
    cfg.training_arguments.audit_data_seed = phase_seed
    cfg.dataloaders.loader_seed = phase_seed
    cfg.training_arguments.adaptive_lambda.enabled = name == "search"
    cfg.model.lambda_coef = float(config.model.lambda_coef) if name == "search" else 0.0
    OmegaConf.update(cfg, "channel_pruning", {"enabled": name != "search", "mode": "explicit",
                     "structural": name != "search", "mask": mask}, merge=False, force_add=True)
    _configure_run_history(cfg, root / name)
    cfg.mlflow.enabled = False
    cfg.run_history.run_name = name
    return cfg


def _run_phase(config, root, state, persist, name, epochs, source, mask, reference=None):
    cfg = _phase_config(config, root, name, epochs, mask)
    source_state = {key: tensor.detach().cpu().clone() for key, tensor in source.state_dict().items()}
    source_hash = state_hash(source_state)
    offset = int(state["global_epochs_completed"])
    data, split = _data(cfg)
    if split != state["split_indices_hash"]:
        raise ValueError("Dataset split changed across phases")
    steps = len(data.train_dataloader)
    del data
    phase = {"name": name, "status": "running", "epochs_allocated": int(epochs),
             "epochs_completed": 0, "initial_state_hash": source_hash,
             "seed": int(cfg.seed), "data_epoch_offset": 0, "adaptive_epoch_offset": 0,
             "optimizer_state_reused": False, "scheduler": "single_continuous_cosine",
             "scheduler_T_max": int(epochs), "adaptive_controller_reused": False,
             "regularization_active": name == "search"}
    state["stages"].append(phase)
    persist()

    def initialize(model):
        model.load_state_dict(source_state, strict=True)
        if state_hash(model.state_dict()) != source_hash:
            raise ValueError("Phase initializer state changed")
        if name != "search":
            assert_fresh_compact(model)

    def record(epoch, train, valid, model, optimizer, history):
        with history.history_path.open(newline="") as stream:
            logged = list(csv.DictReader(stream))[-1]
        if int(logged["epoch"]) != epoch:
            raise ValueError("Phase callback/history epoch mismatch")
        used, following = float(logged["lambda_used"]), float(logged["lambda_next"])
        if not all(math.isfinite(x) and x >= 0 for x in (used, following)):
            raise ValueError("Non-finite adaptive lambda feedback")
        action = logged.get("adaptive_lambda_action", "")
        if name == "search" and (used <= 0 or following <= 0 or not action
                                  or not logged.get("valid_average_zero_prob")):
            raise ValueError("Search is missing live adaptive lambda/gate feedback")
        if name != "search" and (used != 0 or following != 0 or gates(model)):
            raise ValueError("Scratch/dense phase must train without gates or lambda")
        for metric in ("valid_accuracy", "valid_ce_loss"):
            if not math.isfinite(float(valid[metric])):
                raise FloatingPointError(f"Non-finite {metric}")
        global_epoch = offset + int(epoch)
        row = {"global_epoch": global_epoch, "local_epoch": int(epoch), "stage": name,
               "job": root.name, "valid_accuracy": float(valid["valid_accuracy"]),
               "valid_ce_loss": float(valid["valid_ce_loss"]), "lambda_used": used,
               "lambda_next": following, "lr": train["lr"], "mask_hash": mask_hash(mask),
               "adaptive_lambda_action": action if name == "search" else "disabled_no_gates",
               "adaptive_lambda_reason": logged.get("adaptive_lambda_reason", ""),
               "valid_average_zero_prob": logged.get("valid_average_zero_prob", ""),
               "gate_regularization_normalization": "initial_channels",
               "optimizer_steps_total": state["optimizer_steps_total"] + steps,
               "data_epoch": int(epoch), "data_seed": int(cfg.seed)}
        path = root / "global_history.csv"
        exists = path.exists()
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        state["global_epochs_completed"] = global_epoch
        state[{"search": "search_epochs_completed", "scratch": "retrain_epochs_completed",
               "dense": "dense_epochs_completed"}[name]] = int(epoch)
        state["optimizer_steps_total"] += steps
        phase["epochs_completed"] = int(epoch)
        persist()

    kwargs = {"adaptive_reference_by_epoch": reference, "adaptive_epoch_offset": 0} if name == "search" else {}
    result = run_training(cfg, model_initializer=initialize, epoch_end_callback=record, **kwargs)
    if result.get("test_metrics") or not result.get("test_evaluation_disabled"):
        raise RuntimeError("Unexpected test access during training")
    if result["num_epochs_executed"] != epochs or phase["epochs_completed"] != epochs:
        raise RuntimeError("Incomplete phase: refusing to claim the full training budget")
    selected_path = Path(result["run_dir"]) / "checkpoints" / "best.pt"
    payload = torch.load(selected_path, map_location="cpu", weights_only=True)
    if int(payload["epoch"]) != int(result["best_epoch"]):
        raise ValueError("Selected best checkpoint epoch mismatch")
    _check_finite_state(payload["model_state_dict"])
    if name == "search" and not isinstance(payload.get("extra_state", {}).get("adaptive_lambda_state"), dict):
        raise ValueError("Selected search checkpoint has no adaptive controller state")
    selected = deepcopy(source).cpu()
    selected.load_state_dict(payload["model_state_dict"], strict=True)
    phase.update(status="completed", run_dir=str(result["run_dir"]), best_epoch=int(payload["epoch"]),
                 best_metric=float(result["best_metric_value"]), selected_checkpoint=str(selected_path),
                 selected_checkpoint_sha256=_sha(selected_path), selected_state_hash=state_hash(selected.state_dict()))
    persist()
    return selected, payload, selected_path


def _setup(config, output_root, kind, allow_short_run):
    total = validate_config(config, allow_short_run=allow_short_run)
    root = Path(output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty one-shot run: {root}")
    root.mkdir(parents=True, exist_ok=True)
    state = {"protocol": PROTOCOL, "kind": kind, "status": "running", "seed": int(config.seed),
             "reinit_seed": int(config.one_shot_pruning.reinit_seed),
             "total_epochs_allocated": int(config.one_shot_pruning.search_epochs) if kind == "dense_reference" else total,
             "global_epochs_completed": 0, "search_epochs_completed": 0, "retrain_epochs_completed": 0,
             "dense_epochs_completed": 0, "optimizer_steps_total": 0, "stages": [],
             "accepted_mask": {}, "accepted_mask_hash": mask_hash({}), "test_evaluated": False,
             "training_signature": _training_signature(config), "schedule": "single_continuous_cosine",
             "gate_regularization_normalization": "initial_channels", "initial_gate_channels": {},
             "smoke_short_run": bool(allow_short_run), "test_comparisons": "exploratory",
             "search_weights_reused_for_scratch": False, "search_bn_reused_for_scratch": False,
             "search_optimizer_reused_for_scratch": False, "search_controller_reused_for_scratch": False}
    OmegaConf.save(config, root / "resolved_config.yaml", resolve=True)
    state["resolved_config_sha256"] = _sha(root / "resolved_config.yaml")
    def persist():
        write_json(root / "one_shot_state.json", state)
    persist()
    return root, state, persist


def _save_deployment(root, state, model, provenance):
    model.cpu()
    payload = {"protocol": PROTOCOL, "kind": state["kind"], "model_state_dict": model.state_dict(),
               "pruning_mask": state["accepted_mask"], "mask_hash": state["accepted_mask_hash"],
               "model_state_hash": state_hash(model.state_dict()), "validation": state["validation"],
               "global_epochs_consumed": state["global_epochs_completed"], "provenance": provenance,
               "common_init_hash": state["common_init_hash"], "scratch_init_hash": state.get("scratch_init_hash"),
               "gate_regularization_normalization": "initial_channels",
               "initial_gate_channels": state["initial_gate_channels"]}
    _atomic_torch_save(payload, root / "deployment.pt")
    state["deployment_sha256"] = _sha(root / "deployment.pt")
    state["deployment_model_state_hash"] = payload["model_state_hash"]


def _fail(state, persist, exc):
    state.update(status="interrupted" if isinstance(exc, (TrainingInterrupted, KeyboardInterrupt)) else "failed",
                 error=str(exc), partial_epoch_not_counted=True)
    for phase in state["stages"]:
        if phase["status"] == "running":
            phase.update(status=state["status"], partial_epoch_not_counted=True)
    persist()


def _require_device(config):
    if str(config.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU training")


def run_dense_reference(config, output_root, *, allow_short_run=False):
    """Train one ungated, continuous reference; never called implicitly by search."""
    root, state, persist = _setup(config, output_root, "dense_reference", allow_short_run)
    with cooperative_signals():
        try:
            _require_device(config)
            seed = int(config.seed)
            set_random_seed(seed)
            plain_cfg = deepcopy(config.model)
            plain_cfg.lambda_coef = 0.0
            plain_cfg.backbone.resnet_block = {"_target_": "net_complexity.models.resnet.Bottleneck", "_partial_": True}
            plain = instantiate(plain_cfg)
            initial = {key: value.detach().cpu().clone() for key, value in plain.state_dict().items()}
            common_hash = state_hash(initial)
            _atomic_torch_save({"model_state_dict": initial, "trained_epochs": 0, "seed": seed,
                                "model_state_hash": common_hash}, root / "initializer.pt")
            del plain
            dense = _fresh_compact(config, {}, seed)
            _load_plain_initial(dense, initial)
            assert_fresh_compact(dense)
            data, split = _data(config)
            sample = next(iter(data.valid_dataloader))
            state.update(common_init_hash=common_hash, initial_checkpoint_sha256=_sha(root / "initializer.pt"),
                         split_indices_hash=split, initial_cost=deployment_cost(dense, image_shape=sample[0].shape[1:]))
            persist()
            selected, _, path = _run_phase(config, root, state, persist, "dense",
                                          int(config.one_shot_pruning.search_epochs), dense, {})
            state.update(validation=evaluate_deployment(selected, data.valid_dataloader, str(config.device)),
                         final_cost=deployment_cost(selected, image_shape=sample[0].shape[1:]),
                         reference_history_sha256=_sha(root / "global_history.csv"))
            _save_deployment(root, state, selected, {"phase": "dense", "selected_checkpoint": str(path),
                                                    "initializer": "initializer.pt", "trained_from_scratch": True})
            state["status"] = "completed"
            persist()
            return state
        except (Exception, KeyboardInterrupt) as exc:
            _fail(state, persist, exc)
            raise


def run_one_shot_pruning(config, output_root, *, allow_short_run=False):
    """Search → one raw learned mask → discard weights → random compact retrain."""
    root, state, persist = _setup(config, output_root, "one_shot_reinit", allow_short_run)
    with cooperative_signals():
        try:
            _require_device(config)
            c = config.one_shot_pruning
            initial_path, reference_path = Path(str(c.initial_checkpoint)).resolve(), Path(str(c.reference_history)).resolve()
            reference_state = validate_dense_reference(reference_path.parent, config, allow_short_run=allow_short_run)
            if initial_path != Path(reference_state["initial_checkpoint"]) or reference_path != Path(reference_state["reference_history"]):
                raise ValueError("Initializer/reference must belong to the verified dense experiment")
            initial = _load_initial(initial_path, int(config.seed))
            reference = load_adaptive_reference(reference_path, int(c.search_epochs))
            set_random_seed(int(config.seed))
            carrier = instantiate(config.model)
            _load_plain_initial(carrier, initial["model_state_dict"])
            selectors = gates(carrier)
            if not selectors or any(g.regularization_normalization != "initial_channels" for g in selectors.values()):
                raise ValueError("Search requires initial_channels-normalized channel gates")
            data, split = _data(config)
            if split != reference_state["split_indices_hash"]:
                raise ValueError("Search/dense dataset splits differ")
            if reference_state["optimizer_steps_total"] != len(data.train_dataloader) * int(c.search_epochs):
                raise ValueError("Search/dense optimizer steps per epoch differ")
            sample = next(iter(data.valid_dataloader))
            all_open = _fresh_compact(config, {}, int(c.reinit_seed))
            state.update(common_init_hash=initial["model_state_hash"],
                         initial_checkpoint=str(initial_path), initial_checkpoint_sha256=_sha(initial_path),
                         reference_history=str(reference_path), reference_history_sha256=_sha(reference_path),
                         dense_reference_epochs=int(c.search_epochs), dense_reference_cost_included_in_model_epochs=False,
                         initial_gate_channels={name: int(g.logits.shape[0]) for name, g in selectors.items()},
                         split_indices_hash=split, initial_cost=deployment_cost(all_open, image_shape=sample[0].shape[1:]))
            del all_open
            persist()
            selected, payload, selected_path = _run_phase(config, root, state, persist, "search",
                                                         int(c.search_epochs), carrier, {}, reference)
            del carrier, selectors, initial
            mask, decision = mask_from_raw_logits(selected, float(c.mask_threshold))
            decision.update(selected_checkpoint=str(selected_path), selected_checkpoint_epoch=int(payload["epoch"]),
                            selected_checkpoint_sha256=_sha(selected_path), selected_state_hash=state_hash(selected.state_dict()),
                            selected_validation_may_include_temporary_open_bias=True)
            write_json(root / "mask_selection.json", decision)
            state["mask_selection"] = decision
            persist()
            if decision["collapsed_boundaries"]:
                raise ValueError("Learned mask closed an entire boundary; no floor rescue or budget substitution: "
                                 + ", ".join(decision["collapsed_boundaries"]))
            validate_mask(selected, mask)
            # A disposable transfer verifies slicing correctness, not a warm start.
            disposable = _fresh_compact(config, mask, int(c.reinit_seed))
            transfer_gated_weights_to_structural(selected, disposable)
            decision["equivalence"] = committed_equivalence(selected, disposable, mask, sample, str(config.device),
                                                            report_path=root / "equivalence.json")
            decision["canonical_committed_validation"] = evaluate_deployment(disposable, data.valid_dataloader, str(config.device))
            decision["disposable_trained_state_hash"] = state_hash(disposable.state_dict())
            decision["disposable_physical_parameters"] = sum(p.numel() for p in disposable.parameters())
            del selected, disposable, payload
            check_stop()
            scratch = _fresh_compact(config, mask, int(c.reinit_seed))
            initial_scratch_hash = state_hash(scratch.state_dict())
            if initial_scratch_hash == decision["disposable_trained_state_hash"]:
                raise ValueError("Fresh scratch initialization unexpectedly equals the trained compact state")
            scratch_payload = {"protocol": PROTOCOL, "model_state_dict": scratch.state_dict(),
                               "model_state_hash": initial_scratch_hash, "seed": int(c.reinit_seed),
                               "trained_epochs": 0, "pruning_mask": mask, "mask_hash": mask_hash(mask),
                               "source": "fresh_compact_constructor", "batchnorm_fresh": True,
                               "contains_gates": False, "search_state_reused": False}
            _atomic_torch_save(scratch_payload, root / "scratch_initializer.pt")
            compact_cost = deployment_cost(scratch, image_shape=sample[0].shape[1:])
            if compact_cost["physical_total_parameters"] != decision["disposable_physical_parameters"]:
                raise ValueError("Scratch topology differs from the verified extracted topology")
            initial_count = state["initial_cost"]["physical_total_parameters"]
            final_count = compact_cost["physical_total_parameters"]
            decision.update(status="extracted", physical_parameters_before=initial_count,
                            physical_parameters_after=final_count, compression_occurred=final_count < initial_count,
                            removed_parameter_fraction=1 - final_count / initial_count)
            state.update(accepted_mask=mask, accepted_mask_hash=mask_hash(mask), scratch_init_hash=initial_scratch_hash,
                         scratch_initializer_sha256=_sha(root / "scratch_initializer.pt"),
                         physical_prune_events=1, compression_occurred=final_count < initial_count,
                         removed_parameter_fraction=1 - final_count / initial_count)
            write_json(root / "mask_selection.json", decision)
            persist()
            print(f"[one-shot] Physical topology: {initial_count:,} -> {final_count:,} parameters; "
                  f"new random initialization seed={int(c.reinit_seed)}; no search weights reused.", flush=True)
            selected, _, path = _run_phase(config, root, state, persist, "scratch", int(c.retrain_epochs), scratch, mask)
            state.update(validation=evaluate_deployment(selected, data.valid_dataloader, str(config.device)),
                         final_cost=deployment_cost(selected, image_shape=sample[0].shape[1:]))
            if state["global_epochs_completed"] != state["total_epochs_allocated"]:
                raise ValueError("Total completed epoch budget mismatch")
            _save_deployment(root, state, selected, {"phase": "scratch", "selected_checkpoint": str(path),
                                                    "initializer": "scratch_initializer.pt", "trained_from_scratch": True,
                                                    "search_state_reused": False, "reinit_seed": int(c.reinit_seed)})
            state["status"] = "completed"
            persist()
            return state
        except (Exception, KeyboardInterrupt) as exc:
            _fail(state, persist, exc)
            raise
