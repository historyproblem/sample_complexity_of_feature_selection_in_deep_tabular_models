"""Audited physical pruning: historical fixed pilot and opt-in adaptive protocol.

All decisions come from the selected checkpoint. Transactions are evaluated on
validation only; rejecting one stops further pruning and spends only the remaining
epoch allowance on the previous accepted topology.
"""
from __future__ import annotations

import csv
import hashlib
import math
import shutil
from copy import deepcopy
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from net_complexity.models.channel_pruning import (
    apply_channel_mask, build_structurally_pruned_model_from_config,
    transfer_gated_weights_to_structural, transfer_structural_weights_to_gated,
)
from net_complexity.models.pruning_budget import gates, select_by_budget, validate_mask
from .cyclic_aig import _configure_run_history, _set_num_epochs
from .engine import run_training, _set_model_lambda_coef
from .interruption import TrainingInterrupted, check_stop, cooperative_signals
from .pruning_measurement import (
    calibrate_bn, deployment_cost, evaluate_deployment, mask_hash, state_hash, write_json,
)

FIXED_PRUNING_PILOT_VERSION = 1
ADAPTIVE_PRUNING_PILOT_VERSION = 2
ADAPTIVE_PROTOCOL = "adaptive_lambda_v1"


def is_adaptive(config):
    return config.cyclic_channel_pruning.audit_protocol == ADAPTIVE_PROTOCOL


def gate_regularization_normalization(config):
    mode = OmegaConf.select(config, "model.backbone.resnet_block.regularization_normalization",
                            default="enabled_channels")
    if mode not in ("enabled_channels", "initial_channels"):
        raise ValueError("Unsupported gate regularization normalization.")
    return mode


def load_adaptive_reference(path, total_epochs):
    """Load an explicit validation-only reference; never train a hidden baseline."""
    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"global_epoch", "valid_accuracy"}.issubset(reader.fieldnames or []):
            raise ValueError("Adaptive reference requires global_epoch and valid_accuracy columns.")
        result = {}
        for row in reader:
            epoch, accuracy = int(row["global_epoch"]), float(row["valid_accuracy"])
            if epoch in result or epoch < 1 or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise ValueError("Invalid/duplicate adaptive validation reference row.")
            result[epoch] = accuracy
    if not set(range(1, total_epochs + 1)).issubset(result):
        raise ValueError("Adaptive validation reference does not cover the full training budget.")
    return result


def validate_config(config):
    c = config.cyclic_channel_pruning
    adaptive = is_adaptive(config)
    normalization = gate_regularization_normalization(config)
    if not adaptive and normalization != "enabled_channels":
        raise ValueError("Historical fixed pilot requires enabled_channels normalization.")
    supported = {
        "enabled", "audit_protocol", "max_cycles", "stop_on_convergence",
        "gumbel_epochs", "recovery_epochs", "final_epochs", "drop_mode",
        "max_param_fraction", "min_keep_ratio", "weight_handoff", "commit_guard",
        "ranking",
    }
    if adaptive:
        supported.add("adaptive_reference_history")
    unknown = set(c) - supported
    if unknown:
        raise ValueError(f"Unsupported pilot options: {sorted(unknown)}")
    required = {
        "training_arguments.evaluate_test": False,
        "training_arguments.adaptive_lambda.enabled": adaptive,
        "training_arguments.lambda_warmup.enabled": False,
        "training_arguments.batchnorm_recalibration.enabled": False,
        "model.entropy_regularization_coef": 0.0,
        "model.entropy_regularization": "plus_negative_entropy",
        "optimizer.gate_weight_decay_scale": 0.0,
        "cyclic_channel_pruning.stop_on_convergence": False,
        "cyclic_channel_pruning.drop_mode": "param_budget",
        "cyclic_channel_pruning.weight_handoff.enabled": True,
        "cyclic_channel_pruning.weight_handoff.checkpoint_name": "best.pt",
        "cyclic_channel_pruning.commit_guard.enabled": True,
        "model.backbone.resnet_block.train_gate_mode": "ste_hard",
        "model.backbone.resnet_block.eval_gate_mode": "deterministic_hard",
    }
    for key, expected in required.items():
        if OmegaConf.select(config, key) != expected:
            raise ValueError(f"{'Adaptive' if adaptive else 'Fixed'} pilot requires {key}={expected!r}.")
    if adaptive:
        if not math.isfinite(float(config.model.lambda_coef)) or float(config.model.lambda_coef) <= 0:
            raise ValueError("Adaptive pruning requires a positive finite initial lambda.")
        if OmegaConf.select(config, "training_arguments.adaptive_lambda.baseline_history_dir") not in (None, ""):
            raise ValueError("Adaptive pilot requires an explicit reference, not automatic baseline training.")
        if not isinstance(c.get("adaptive_reference_history"), str) or not c.adaptive_reference_history.strip():
            raise ValueError("Adaptive pilot requires adaptive_reference_history.")
    elif float(config.model.lambda_coef) not in (0.0, 0.001):
        raise ValueError("Fixed pilot supports lambda=0 or 0.001 only.")
    if float(config.model.lambda_coef) == 0 and float(c.max_param_fraction) != 0:
        raise ValueError("Dense control must have zero pruning budget.")
    if str(getattr(c, "ranking", "learned")) not in ("learned", "random"):
        raise ValueError("Unsupported channel ranking.")
    if set(c.weight_handoff) != {"enabled", "checkpoint_name", "initial_checkpoint"}:
        raise ValueError("Unsupported/incomplete weight_handoff.")
    if OmegaConf.select(config, "run_history.monitor") != "valid_accuracy":
        raise ValueError("Checkpoint selection must use validation accuracy.")
    if OmegaConf.select(config, "run_history.secondary_monitor") != "valid_ce_loss":
        raise ValueError("Checkpoint ties must use weighted validation CE.")
    if OmegaConf.select(config, "training_arguments.collapse_guard.enabled", default=False):
        raise ValueError("Use the transactional guard, not legacy early collapse stopping.")
    if not 0 <= float(c.max_param_fraction) < 1 or not 0 < float(c.min_keep_ratio) <= 1:
        raise ValueError("Invalid budget/floor.")
    for name in ("max_cycles", "gumbel_epochs", "recovery_epochs", "final_epochs"):
        if type(c[name]) is not int or c[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if OmegaConf.select(config, "training_arguments.early_stopping.enabled", default=False):
        raise ValueError("Fixed budget pilot does not support early stopping.")
    if OmegaConf.select(config, "training_arguments.gate_mode_schedule.enabled", default=False):
        raise ValueError("Fixed gate modes required.")
    guard = c.commit_guard
    if set(guard) != {"enabled", "train_bn_calibration_batches",
                      "max_immediate_accuracy_drop", "max_recovered_accuracy_drop", "on_reject"}:
        raise ValueError("Unsupported/incomplete commit_guard.")
    if guard.on_reject != "continue_previous_graph_within_budget":
        raise ValueError("Unsupported rollback policy.")
    for name in ("max_immediate_accuracy_drop", "max_recovered_accuracy_drop"):
        if not 0 <= guard[name] <= 1:
            raise ValueError(f"Invalid {name}.")
    if guard.train_bn_calibration_batches < 0:
        raise ValueError("Negative calibration budget.")
    total = c.max_cycles * c.gumbel_epochs + (c.max_cycles - 1) * c.recovery_epochs + c.final_epochs
    if adaptive and total > 150:
        raise ValueError("Adaptive pruning search + all recovery must fit within 150 epochs.")
    return total


def build_structural(config, carrier, mask):
    validate_mask(carrier, mask)
    cfg = deepcopy(config)
    cfg.model.lambda_coef = 0.0
    pruning = OmegaConf.create({"enabled": True, "mode": "explicit", "structural": True, "mask": mask})
    model = build_structurally_pruned_model_from_config(cfg, pruning)
    transfer_gated_weights_to_structural(carrier, model)
    return model


def selected_model(config, result):
    path = Path(result["run_dir"]) / "checkpoints" / "best.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if int(payload["epoch"]) != int(result["best_epoch"]):
        raise ValueError("Decision epoch differs from selected checkpoint epoch.")
    model = instantiate(config.model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    extra = payload.get("extra_state", {})
    if "model_lambda_coef" in extra:
        bypass = extra.get("gumbel_bypass_enabled")
        _set_model_lambda_coef(model, float(extra["model_lambda_coef"]),
                               bypass_gumbel=None if bypass is None else bool(bypass))
    return model, payload, path


def _equivalence_stats(actual, expected, *, rtol, atol):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise FloatingPointError("Non-finite logits in committed equivalence check.")
    if actual.shape != expected.shape:
        raise AssertionError("Committed equivalence logit shapes differ.")
    error = (actual - expected).abs()
    return {"max_abs_error": float(error.max()), "logit_count": actual.numel(),
            "mismatched_logits": int((error > atol + rtol * expected.abs()).sum()),
            "prediction_disagreements": int((actual.argmax(-1) != expected.argmax(-1)).sum()),
            "rtol": rtol, "atol": atol}


@torch.no_grad()
def committed_equivalence(carrier, structural, mask, sample, device, *, report_path=None):
    """Allow small FP32 roundoff only if an independent FP64 check also passes.

    Narrow convolutions can use different accumulation orders. The fallback
    checks the SAME entire validation batch on CPU in double precision;
    small FP32 differences alone are not sufficient to accept the transfer.
    """
    validate_mask(carrier, mask)
    clone = deepcopy(carrier).to(device).eval()
    for gate in gates(clone).values():
        gate.channel_mask.fill_(1)
        gate.set_bypass(True)
    apply_channel_mask(clone, mask)
    report = {"status": "failed", "device": str(device), "policy": "fp32_then_bounded_fp64_v1"}
    try:
        structural.to(device).eval()
        x, y = (t.to(device) for t in sample)
        actual, expected = clone(x, y).logits, structural(x, y).logits
        report["fp32"] = _equivalence_stats(actual, expected, rtol=1e-4, atol=1e-5)
        if report["fp32"]["mismatched_logits"] == 0:
            report["status"] = "passed_fp32"
            return report
        # Large FP32 discrepancies are not explained away by a second backend.
        report["fp32_ceiling"] = _equivalence_stats(actual, expected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        print(f"[equivalence] FP32 max error={report['fp32']['max_abs_error']:.3g}; "
              "checking the full batch in CPU float64.", flush=True)
        clone.cpu().double()
        reference = deepcopy(structural).cpu().double().eval()
        x64, y64 = sample[0].cpu().double(), sample[1].cpu()
        double_actual, double_expected = [], []
        for start in range(0, len(x64), 8):
            check_stop()
            batch = x64[start:start + 8], y64[start:start + 8]
            double_actual.append(clone(*batch).logits)
            double_expected.append(reference(*batch).logits)
        actual64, expected64 = torch.cat(double_actual), torch.cat(double_expected)
        report["fp64"] = _equivalence_stats(actual64, expected64, rtol=1e-8, atol=1e-9)
        torch.testing.assert_close(actual64, expected64, rtol=1e-8, atol=1e-9)
        report["status"] = "passed_fp64_fallback"
        print(f"[equivalence] FP64 passed; max error={report['fp64']['max_abs_error']:.3g}.", flush=True)
        return report
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        structural.cpu()
        if report_path is not None:
            write_json(Path(report_path), report)


def _run(config, output_root, *, resume_search_from=None):
    total_epochs = validate_config(config)
    adaptive = is_adaptive(config)
    if adaptive and resume_search_from is not None:
        raise ValueError("Legacy fixed search resume is not an adaptive controller resume.")
    reference = (load_adaptive_reference(config.cyclic_channel_pruning.adaptive_reference_history,
                                         total_epochs) if adaptive else None)
    controller_state = None
    controller_handoffs = []
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "pilot_state.json").exists():
        raise FileExistsError("Refusing reuse of an existing pilot run.")
    c = config.cyclic_channel_pruning
    device = str(config.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU overnight run.")
    seed = int(config.seed)
    # The shared initializer contains only backbone weights, not gate logits.
    # Seed BEFORE creating the carrier; engine seeding happens too late for it.
    from .randomness import set_random_seed
    set_random_seed(seed)
    init_path = Path(str(c.weight_handoff.initial_checkpoint))
    initial = torch.load(init_path, map_location="cpu", weights_only=True)
    if initial.get("trained_epochs") != 0 or initial.get("seed") != seed:
        raise ValueError("Pilot requires the shared zero-trained-epoch initializer with matching seed.")
    common_hash = state_hash(initial["model_state_dict"])
    carrier = instantiate(config.model)
    normalization = gate_regularization_normalization(config)
    initial_gate_channels = {name: int(gate.logits.shape[0]) for name, gate in gates(carrier).items()}
    if any(getattr(gate, "regularization_normalization", "enabled_channels") != normalization
           for gate in gates(carrier).values()):
        raise ValueError("Constructed gate normalization differs from requested configuration.")
    incompatible = carrier.load_state_dict(initial["model_state_dict"], strict=False)
    if incompatible.unexpected_keys or any("gumbel_layer" not in k for k in incompatible.missing_keys):
        raise ValueError(f"Incompatible shared initializer: {incompatible}")
    data = instantiate(config.dataloaders, include_test=False, loader_seed=seed)
    if str(config.dataloaders._target_).endswith("ClassicCVDataloaders"):
        if (len(data.train_dataloader.dataset), len(data.valid_dataloader.dataset)) != (45000, 5000):
            raise ValueError("CIFAR10 pilot requires the matched 45000/5000 train/valid split.")
    if (device.startswith("cuda") and data.train_dataloader.num_workers == 0
            and str(config.dataloaders._target_).endswith("ClassicCVDataloaders")):
        raise ValueError("GPU pilot requires data workers to isolate augmentation RNG from gate RNG.")
    # Validation is the only holdout constructed by this runner.
    sample = next(iter(data.valid_dataloader))
    image_shape = tuple(sample[0].shape[1:])
    mask = {}
    accepted = build_structural(config, carrier, mask)
    initial_cost = deployment_cost(accepted, image_shape=image_shape)
    current_metrics = None
    global_epoch = 0
    stages, decisions = [], []
    split_hash = mask_hash({
        split: list(getattr(loader.dataset, "indices", range(len(loader.dataset))))
        for split, loader in (("train", data.train_dataloader), ("valid", data.valid_dataloader))
    })
    state = {"status": "running", "pilot_version": (ADAPTIVE_PRUNING_PILOT_VERSION if adaptive
                                                     else FIXED_PRUNING_PILOT_VERSION),
             "total_epochs_allocated": total_epochs, "common_init_hash": common_hash,
             "split_indices_hash": split_hash, "seed": seed, "test_evaluated": False,
             "initial_cost": initial_cost, "stages": stages, "decisions": decisions}
    if adaptive:
        state.update(protocol=ADAPTIVE_PROTOCOL, adaptive_lambda_enabled=True,
                     gate_regularization_normalization=normalization,
                     initial_gate_channels=initial_gate_channels,
                     adaptive_reference_history=str(Path(c.adaptive_reference_history).resolve()),
                     adaptive_reference_sha256=hashlib.sha256(Path(c.adaptive_reference_history).read_bytes()).hexdigest(),
                     adaptive_controller_handoffs=controller_handoffs,
                     controller_handoff_policy="selected_search_checkpoint_post_update; held_during_structural_recovery")
    if resume_search_from is not None:
        from .pruning_resume import load_completed_search
        resumed = load_completed_search(config, resume_search_from, common_hash=common_hash,
                                        split_hash=split_hash, steps_per_epoch=len(data.train_dataloader))
        stages.append(resumed["stage"])
        global_epoch = int(resumed["metadata"]["epochs_reused"])
        state["resume"] = resumed["metadata"]
        shutil.copyfile(resumed["history_path"], output_root / "global_history.csv")
        print(f"[resume] Reusing {global_epoch} completed search epochs from {resume_search_from}; "
              "continuing at the pruning decision, not retraining search.", flush=True)
    OmegaConf.save(config, output_root / "resolved_config.yaml", resolve=True)

    def persist():
        state.update(global_epochs_completed=global_epoch, accepted_mask=mask,
                     accepted_mask_hash=mask_hash(mask))
        write_json(output_root / "pilot_state.json", state)

    def stage(name, epochs, source, current_mask, structural):
        nonlocal global_epoch, controller_state
        check_stop()
        cfg = deepcopy(config)
        _set_num_epochs(cfg, epochs)
        cfg.training_arguments.global_epoch_offset = global_epoch
        cfg.training_arguments.audit_data_seed = seed
        cfg.dataloaders.loader_seed = seed
        cfg.model.lambda_coef = 0.0 if structural else config.model.lambda_coef
        adaptive_kwargs = {}
        if adaptive:
            # The physically narrowed graph has no gate penalty. The selected
            # search controller is held, not reset, until the next gated phase.
            cfg.training_arguments.adaptive_lambda.enabled = not structural
            if not structural:
                if {key: int(gate.logits.shape[0]) for key, gate in gates(source).items()} != initial_gate_channels:
                    raise ValueError("Search carrier changed the original gate normalization widths.")
                if any(gate.regularization_normalization != normalization for gate in gates(source).values()):
                    raise ValueError("Gate normalization changed across search/recovery handoff.")
                adaptive_kwargs = {"adaptive_lambda_state": controller_state,
                                   "adaptive_epoch_offset": global_epoch,
                                   "adaptive_reference_by_epoch": reference}
        OmegaConf.update(cfg, "channel_pruning", {
            "enabled": True, "mode": "explicit", "structural": structural, "mask": current_mask,
        }, merge=False, force_add=True)
        _configure_run_history(cfg, output_root / name)
        cfg.mlflow.enabled = False
        cfg.run_history.run_name = name
        source_state = {k: v.detach().cpu().clone() for k, v in source.state_dict().items()}
        offset = global_epoch

        def initialize(model):
            model.load_state_dict(source_state, strict=True)

        def record(epoch, train, valid, model, optimizer, history):
            nonlocal global_epoch
            global_epoch = offset + epoch
            row = {"global_epoch": global_epoch, "local_epoch": epoch, "stage": name,
                   "valid_accuracy": valid["valid_accuracy"], "valid_ce_loss": valid["valid_ce_loss"],
                   "lambda_used": float(cfg.model.lambda_coef), "lambda_next": float(cfg.model.lambda_coef),
                   "lr": train["lr"], "mask_hash": mask_hash(current_mask),
                   "optimizer_steps_total": global_epoch * len(data.train_dataloader)}
            if adaptive:
                with history.history_path.open(newline="") as stream:
                    logged = list(csv.DictReader(stream))[-1]
                if int(logged["epoch"]) != epoch:
                    raise ValueError("Adaptive history epoch differs from current completed epoch.")
                used, following = float(logged["lambda_used"]), float(logged["lambda_next"])
                if not all(math.isfinite(value) and value >= 0 for value in (used, following)):
                    raise ValueError("Non-finite adaptive lambda in epoch history.")
                action = logged.get("adaptive_lambda_action", "")
                if not structural and (used <= 0 or following <= 0 or not action
                                       or not logged.get("valid_average_zero_prob")):
                    raise ValueError("Adaptive search is missing live lambda/gate feedback.")
                row.update(lambda_used=used, lambda_next=following,
                           job=output_root.name,
                           gate_regularization_normalization=normalization,
                           initial_gate_channels=sum(initial_gate_channels.values()),
                           remaining_gate_channels=sum(width - len(current_mask.get(key, []))
                                                       for key, width in initial_gate_channels.items()),
                           adaptive_lambda_action=action if not structural else "held_no_structural_gates",
                           adaptive_lambda_reason=logged.get("adaptive_lambda_reason", ""),
                           adaptive_lambda_step=logged.get("adaptive_lambda_step", ""),
                           valid_average_zero_prob=logged.get("valid_average_zero_prob", ""))
            path = output_root / "global_history.csv"
            exists = path.exists()
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=row.keys())
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            persist()

        result = run_training(cfg, model_initializer=initialize, epoch_end_callback=record, **adaptive_kwargs)
        if result.get("test_metrics") or not result.get("test_evaluation_disabled"):
            raise RuntimeError("Unexpected test access in training result.")
        if result["num_epochs_executed"] != epochs:
            raise RuntimeError("Incomplete stage; refusing to claim full epoch budget.")
        stages.append({"name": name, "epochs": epochs, "run_dir": result["run_dir"],
                       "best_epoch": result["best_epoch"], "best_metric": result["best_metric_value"]})
        if adaptive:
            stages[-1].update(gate_regularization_normalization=normalization,
                              regularization_active=not structural)
        checkpoint = torch.load(Path(result["run_dir"]) / "checkpoints" / "best.pt",
                                map_location="cpu", weights_only=True)
        if adaptive and not structural:
            extra = checkpoint.get("extra_state", {})
            if not isinstance(extra.get("adaptive_lambda_state"), dict):
                raise ValueError("Selected search checkpoint has no adaptive controller state.")
            controller_state = deepcopy(extra["adaptive_lambda_state"])
            controller_handoffs.append({"stage": name, "selected_epoch": int(checkpoint["epoch"]),
                                        "selected_global_epoch": offset + int(checkpoint["epoch"]),
                                        "next_search_global_offset": global_epoch,
                                        "lambda_used_at_selected_epoch": extra["model_lambda_coef"],
                                        "controller_state": controller_state})
            write_json(output_root / "adaptive_controller.json", controller_handoffs[-1])
        model = deepcopy(source).cpu()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model, result

    def save_accepted(model, metrics, provenance):
        payload = {"model_state_dict": model.cpu().state_dict(), "pruning_mask": mask,
                   "mask_hash": mask_hash(mask), "model_state_hash": state_hash(model.state_dict()),
                   "validation": metrics, "provenance": provenance,
                   "global_epochs_consumed": global_epoch, "common_init_hash": common_hash}
        if adaptive:
            payload["gate_regularization_normalization"] = normalization
            payload["initial_gate_channels"] = initial_gate_channels
        temporary = output_root / "deployment.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(output_root / "deployment.pt")

    persist()
    try:
        for cycle in range(c.max_cycles):
            if resume_search_from is not None:
                search_result = {"run_dir": stages[0]["run_dir"], "best_epoch": stages[0]["best_epoch"]}
            else:
                search, search_result = stage(f"cycle_{cycle}_search", c.gumbel_epochs, carrier, mask, False)
            # Re-read best.pt and verify selection epoch; ignore last_valid_metrics.
            search, search_payload, search_path = selected_model(config, search_result)
            if (resume_search_from is not None
                    and state_hash(search_payload["model_state_dict"])
                    != state["resume"]["selected_checkpoint_state_hash"]):
                raise ValueError("Reused search checkpoint changed after resume validation.")
            for name, gate in gates(search).items():
                expected = torch.ones_like(gate.channel_mask)
                expected[mask.get(name, [])] = 0
                if not torch.equal(expected, gate.channel_mask):
                    raise ValueError(f"Selected checkpoint has a different permanent mask: {name}")
            decision_hash = state_hash(search_payload["model_state_dict"])
            candidate_mask, budget = select_by_budget(
                search, mask, float(c.max_param_fraction), float(c.min_keep_ratio),
                ranking=str(getattr(c, "ranking", "learned")), seed=seed + cycle,
            )
            old = build_structural(config, search, mask)
            candidate = build_structural(config, search, candidate_mask)
            equivalence = committed_equivalence(
                search, candidate, candidate_mask, sample, device,
                report_path=output_root / f"cycle_{cycle}_equivalence.json")
            if sum(p.numel() for p in candidate.parameters()) != budget["params_after"]:
                raise AssertionError("Physical count differs from exact budget.")
            if sum(p.numel() for p in old.parameters()) != budget["params_before"]:
                raise AssertionError("Budget denominator differs from current physical count.")
            for model in (old, candidate):
                calibrate_bn(model, data.train_dataloader, device,
                             c.commit_guard.train_bn_calibration_batches, seed + cycle)
            old_metrics = evaluate_deployment(old, data.valid_dataloader, device)
            candidate_metrics = evaluate_deployment(candidate, data.valid_dataloader, device)
            old.cpu()
            candidate.cpu()
            if current_metrics is None:
                accepted, current_metrics = old, old_metrics
                save_accepted(accepted, current_metrics, {"stage": "first_search_open_mask"})
            decision = {"cycle": cycle, "decision_checkpoint": str(search_path),
                        "decision_checkpoint_epoch": search_payload["epoch"],
                        "decision_checkpoint_hash": decision_hash,
                        "candidate_mask": candidate_mask, "candidate_mask_hash": mask_hash(candidate_mask),
                        "budget": budget, "equivalence": equivalence, "old_committed_valid": old_metrics,
                        "candidate_committed_valid": candidate_metrics}
            decisions.append(decision)
            rejected = candidate_metrics["accuracy"] < old_metrics["accuracy"] - c.commit_guard.max_immediate_accuracy_drop
            if rejected:
                decision["status"] = "rejected_before_recovery"
            else:
                recovery_epochs = c.final_epochs if cycle == c.max_cycles - 1 else c.recovery_epochs
                recovered, recovery_result = stage(f"cycle_{cycle}_recovery", recovery_epochs,
                                                   candidate, candidate_mask, True)
                recovered_metrics = evaluate_deployment(recovered, data.valid_dataloader, device)
                recovered.cpu()
                decision["recovered_valid"] = recovered_metrics
                rejected = recovered_metrics["accuracy"] < current_metrics["accuracy"] - c.commit_guard.max_recovered_accuracy_drop
                if rejected:
                    decision["status"] = "rejected_after_recovery"
                else:
                    decision["status"] = "accepted"
                    mask, accepted, current_metrics = candidate_mask, recovered, recovered_metrics
                    save_accepted(accepted, current_metrics, stages[-1])
                    carrier = search
                    transfer_structural_weights_to_gated(accepted, carrier)
                    apply_channel_mask(carrier, mask)
            persist()
            if rejected:
                # Roll back both topology AND weights. Used search/recovery epochs
                # are still charged; no hidden retries or extra training.
                remaining = total_epochs - global_epoch
                if remaining:
                    fallback, fallback_result = stage("rollback_finetune", remaining, accepted, mask, True)
                    fallback_metrics = evaluate_deployment(fallback, data.valid_dataloader, device)
                    if (fallback_metrics["accuracy"], -fallback_metrics["ce_loss"]) >= (
                            current_metrics["accuracy"], -current_metrics["ce_loss"]):
                        accepted, current_metrics = fallback.cpu(), fallback_metrics
                        save_accepted(accepted, current_metrics, stages[-1])
                break
        assert global_epoch == total_epochs
        final_cost = deployment_cost(accepted, image_shape=image_shape, device=device,
                                     latency=device.startswith("cuda"))
        # A missed budget is explicit even if rollback preserves high accuracy.
        ideal_params = initial_cost["physical_total_parameters"] * (1 - c.max_param_fraction) ** c.max_cycles
        state.update(status="completed", validation=current_metrics, final_cost=final_cost,
                     all_pruning_decisions_accepted=all(d["status"] == "accepted" for d in decisions),
                     ideal_parameter_target=ideal_params,
                     parameter_target_met=final_cost["physical_total_parameters"] <= ideal_params * 1.01,
                     optimizer_steps_total=global_epoch * len(data.train_dataloader))
        persist()
        return state
    except Exception as exc:
        state.update(status="interrupted" if isinstance(exc, TrainingInterrupted) else "failed",
                     error=str(exc), partial_epoch_not_counted=True)
        persist()
        raise


def run_fixed_pruning_pilot(config, output_root, *, resume_search_from=None):
    if is_adaptive(config):
        raise ValueError("Use run_adaptive_pruning_pilot for the adaptive protocol.")
    with cooperative_signals():
        return _run(config, output_root, resume_search_from=resume_search_from)


def run_adaptive_pruning_pilot(config, output_root):
    if not is_adaptive(config):
        raise ValueError("Adaptive entrypoint refuses a fixed-lambda configuration.")
    with cooperative_signals():
        return _run(config, output_root)
