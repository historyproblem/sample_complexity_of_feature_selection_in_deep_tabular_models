"""Accuracy-guided v3 stage orchestration over the shared training engine.

Search costs rank learned, quality-feasible masks; they never prescribe a quota.
Physical recovery holds the selected controller state and starts a fresh optimizer.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import subprocess
import time

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from net_complexity.models.channel_pruning import apply_channel_mask, transfer_structural_weights_to_gated
from net_complexity.models.feature_selection import (
    get_gate_normalization_metadata, validate_gate_normalization_metadata,
    get_gate_regularization_diagnostics,
)
from net_complexity.models.pruning_budget import gates, select_learned_closed
from .cyclic_aig import _configure_run_history, _set_num_epochs
from .engine import run_training
from .interruption import TrainingInterrupted, cooperative_signals, check_stop
from .pruning_audit import build_structural, committed_equivalence, load_adaptive_reference
from .pruning_measurement import (
    calibrate_bn, compare_predictors, deployment_cost, evaluate_deployment, gated_export_equivalence,
    isolated_diagnostic_rng, mask_hash, state_hash, write_json,
)


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def select_checkpoint_records(records, reference_accuracy, hard_drop, *, search=True):
    """Pure stage-end selection; no mutation of model, mask or consumed ledger."""
    threshold = float(reference_accuracy) - float(hard_drop)
    if not records or not math.isfinite(threshold):
        raise ValueError("Selection requires checkpoints and a finite reference.")
    for record in records:
        if not all(math.isfinite(float(record[k])) for k in ("accuracy", "ce_loss", "physical_cost")):
            raise FloatingPointError("Non-finite checkpoint selection metrics/cost.")
    feasible = [record for record in records if record["accuracy"] >= threshold]
    accuracy_key = lambda record: (-record["accuracy"], record["ce_loss"], record["epoch"])
    if search and feasible:
        selected = min(feasible, key=lambda record: (record["physical_cost"], *accuracy_key(record)))
    else:
        selected = min(feasible or records, key=accuracy_key)
    return selected, {"policy": "best_feasible_compact" if search else "best_validation_accuracy",
                      "quality_threshold": threshold, "reference_accuracy": float(reference_accuracy),
                      "no_feasible_search": search and not feasible,
                      "feasible_count": len(feasible), "selected_epoch": selected["epoch"],
                      "trace": records}


def prepare_carry(physical, carrier, mask, metadata):
    """Materialize weight handoff once; preserve every surviving raw gate logit."""
    logits = {name: gate.logits.detach().clone() for name, gate in gates(carrier).items()}
    transfer_structural_weights_to_gated(physical, carrier)
    apply_channel_mask(carrier, mask)
    validate_gate_normalization_metadata(carrier, metadata)
    for name, gate in gates(carrier).items():
        if not torch.equal(gate.logits.detach(), logits[name]):
            raise AssertionError("Carry handoff changed raw gate logits.")
        expected = torch.ones_like(gate.channel_mask)
        expected[mask.get(name, [])] = 0
        if not torch.equal(gate.channel_mask, expected):
            raise AssertionError("Carry handoff restored permanently removed coordinates.")
    return carrier


def _code_provenance():
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)
    try:
        # Include untracked implementation content as well as tracked dirty diff.
        digest = hashlib.sha256(git("diff", "HEAD", "--binary"))
        for name in git("ls-files", "--others", "--exclude-standard").decode().splitlines():
            file = root / name
            if file.is_file() and file.suffix in (".py", ".yaml", ".md"):
                digest.update(name.encode())
                digest.update(file.read_bytes())
        return {"commit": git("rev-parse", "HEAD").decode().strip(), "dirty_diff_sha256": digest.hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_diff_sha256": None, "status": "git_unavailable"}


def run_accuracy_guided_pruning(config, output_root, *, resume_from=None):
    """Execute one validated plan. Full-run recovery from an ambiguous run is refused.

    The shared engine has an explicit same-stage snapshot API. This orchestrator
    does not guess a transaction continuation from a legacy or partial run.
    """
    from .accuracy_guided_config import validate_config_v3, validate_inputs
    from .pruning_resume import validate_epoch_eval_checkpoint
    total = validate_config_v3(config)
    if resume_from is not None:
        raise ValueError("Exact whole-plan resume is unavailable: require a consistent transaction snapshot; "
                         "use the shared engine exact same-stage API for a complete epoch snapshot.")
    inputs = validate_inputs(config)
    c = config.accuracy_guided
    reference = load_adaptive_reference(c.reference.history_path, total)
    reference_sha = hashlib.sha256(Path(c.reference.history_path).read_bytes()).hexdigest()
    output_root = Path(output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Refusing reuse of a nonempty accuracy-guided run directory.")
    output_root.mkdir(parents=True, exist_ok=True)
    config = deepcopy(config)
    OmegaConf.save(config, output_root / "resolved_config.yaml", resolve=True)
    device, seed = str(config.device), int(config.seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    from .randomness import set_random_seed
    set_random_seed(seed)
    carrier = instantiate(config.model)
    initializer = torch.load(c.initializer.path, map_location="cpu", weights_only=True)
    if initializer.get("trained_epochs") != 0 or initializer.get("seed") != seed:
        raise ValueError("Initializer must have trained_epochs=0 and the configured seed.")
    mismatch = carrier.load_state_dict(initializer["model_state_dict"], strict=False)
    if mismatch.unexpected_keys or any("gumbel_layer" not in key for key in mismatch.missing_keys):
        raise ValueError(f"Incompatible zero-epoch backbone initializer: {mismatch}")
    normalization = get_gate_normalization_metadata(carrier)
    if not gates(carrier) or any(g.regularization_normalization != "initial_channels" for g in gates(carrier).values()):
        raise ValueError("Runtime must contain initial_channels gates.")
    data = instantiate(config.dataloaders, include_test=False, loader_seed=seed)
    # Entirely separate iteration/worker streams for validation diagnostics and BN calibration.
    with isolated_diagnostic_rng():
        diagnostic_data = instantiate(config.dataloaders, include_test=False, loader_seed=seed)
        sample = next(iter(diagnostic_data.valid_dataloader))
    if str(config.dataloaders._target_).endswith("ClassicCVDataloaders"):
        if (len(data.train_dataloader.dataset), len(data.valid_dataloader.dataset)) != (45000, 5000):
            raise ValueError("The matched CIFAR split must contain 45000/5000 examples.")
    image_shape = tuple(sample[0].shape[1:])
    split_hash = mask_hash({name: list(getattr(loader.dataset, "indices", range(len(loader.dataset))))
                           for name, loader in (("train", data.train_dataloader), ("valid", data.valid_dataloader))})
    reference_state = json.loads(Path(c.reference.state_path).read_text())
    if reference_state.get("split_indices_hash") != split_hash:
        raise ValueError("Actual runtime split indices differ from the immutable validation reference.")
    accepted_mask, accepted, accepted_metrics = {}, None, None
    accepted_origin = None
    controller_state, pending_rebase = None, None
    ledger = dict(global_training_epoch=0, search_epochs_consumed=0,
                  optimizer_updates=0, consumed_training_examples=0)
    stage_plan = OmegaConf.to_container(c.stage_plan, resolve=True)
    provenance = {**_code_provenance(), "inputs": inputs,
                  "config_sha256": hashlib.sha256(OmegaConf.to_yaml(config, resolve=True).encode()).hexdigest(),
                  "reference_sha256": reference_sha, "initializer_sha256": hashlib.sha256(Path(c.initializer.path).read_bytes()).hexdigest(),
                  "initializer_state_hash": state_hash(initializer["model_state_dict"]),
                  "normalization": normalization, "split_indices_hash": split_hash,
                  "seed": seed, "hardware": platform.platform(), "device": device,
                  "torch_version": str(torch.__version__), "dtype": str(next(carrier.parameters()).dtype)}
    initial_probe_started = time.perf_counter()
    initial_cost = deployment_cost(build_structural(config, carrier, {}), image_shape=image_shape)
    initial_probe_seconds = time.perf_counter() - initial_probe_started
    state = {"protocol": c.protocol, "artifact_type": "physical_ungated", "status": "running",
             "test_evaluated": False, "stage_plan": stage_plan, "stages": [], "decisions": [],
             "transitions": [], "epoch_events": [], "ledger": ledger, "provenance": provenance,
             "initial_cost": initial_cost, "total_epochs_allocated": total,
             "seed": seed, "common_init_hash": provenance["initializer_state_hash"], "split_indices_hash": split_hash,
             "orchestration_overhead": {"validation_forward_examples": 0, "validation_wall_seconds": 0.0,
                "diagnostic_forward_examples": 0, "diagnostic_wall_seconds": 0.0,
                "calibration_forward_examples": 0, "calibration_wall_seconds": 0.0,
                "cost_probe_forward_examples": 1, "cost_probe_wall_seconds": initial_probe_seconds,
                "scope": "orchestration only; per-epoch training/validation timing remains in each shared-engine history"},
             "normalization": "initial_channels", "scaling_contract": "survivor_equivalent_v1"}
    started = time.perf_counter()
    held_alpha = float(config.model.lambda_coef)

    def persist():
        state.update(accepted_mask=accepted_mask, accepted_mask_hash=mask_hash(accepted_mask),
                     global_epochs_completed=ledger["global_training_epoch"],
                     optimizer_steps_total=ledger["optimizer_updates"],
                     wall_seconds=time.perf_counter() - started)
        write_json(output_root / "protocol_state.json", state)

    def evaluate(model):
        started_evaluation = time.perf_counter()
        with isolated_diagnostic_rng(diagnostic_data.valid_dataloader):
            result = evaluate_deployment(model, diagnostic_data.valid_dataloader, device)
        model.cpu()
        state["orchestration_overhead"]["validation_forward_examples"] += result["example_count"]
        state["orchestration_overhead"]["validation_wall_seconds"] += time.perf_counter() - started_evaluation
        return result

    def compare(physical, gated):
        result = compare_predictors(physical, gated, diagnostic_data.valid_dataloader, device)
        state["orchestration_overhead"]["diagnostic_forward_examples"] += result["overhead"]["forward_examples"]
        state["orchestration_overhead"]["diagnostic_wall_seconds"] += result["overhead"]["wall_seconds"]
        return result

    def measure_cost(model, *, device="cpu"):
        start = time.perf_counter()
        with isolated_diagnostic_rng():
            result = deployment_cost(model, image_shape=image_shape, device=device)
        state["orchestration_overhead"]["cost_probe_forward_examples"] += 1
        state["orchestration_overhead"]["cost_probe_wall_seconds"] += time.perf_counter() - start
        return result

    def check_transfer(gated, physical, mask):
        start = time.perf_counter()
        with isolated_diagnostic_rng():
            report = committed_equivalence(gated, physical, mask, sample, device)
        report["forward_examples"] = len(sample[1]) * (4 if "fp64" in report else 2)
        report["wall_seconds"] = time.perf_counter() - start
        state["orchestration_overhead"]["diagnostic_forward_examples"] += report["forward_examples"]
        state["orchestration_overhead"]["diagnostic_wall_seconds"] += report["wall_seconds"]
        return report

    def save_deployment(origin):
        nonlocal accepted_origin
        if origin is not None:
            accepted_origin = deepcopy(origin)
        if accepted_origin is None:
            raise ValueError("Accepted physical deployment lacks selected-checkpoint provenance.")
        state["deployment_origin"] = deepcopy(accepted_origin)
        atomic_checkpoint(output_root / "deployment.pt", {
            "version": 3, "protocol": c.protocol, "artifact_type": "physical_ungated",
            "model_state_dict": accepted.cpu().state_dict(), "model_state_hash": state_hash(accepted.state_dict()),
            "pruning_mask": accepted_mask, "mask_hash": mask_hash(accepted_mask),
            "normalization_metadata": normalization, "validation": accepted_metrics,
            "provenance": {**provenance, "selected_origin": accepted_origin}, "ledger": deepcopy(ledger),
            "global_epochs_consumed": ledger["global_training_epoch"],
            "common_init_hash": provenance["initializer_state_hash"],
            "controller_held": deepcopy(controller_state), "eval_runtime": {"gate_count": 0, "training": False},
        })

    def stage(spec, source, mask, *, structural, rebase=None):
        nonlocal controller_state, held_alpha
        check_stop()
        cfg = deepcopy(config)
        epochs, offset = spec["epochs"], ledger["global_training_epoch"]
        end_epoch = offset + epochs
        actual_training_graph_cost = measure_cost(source)
        stage_started = time.perf_counter()
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        _set_num_epochs(cfg, epochs)
        cfg.training_arguments.global_epoch_offset = offset
        cfg.model.lambda_coef = 0.0 if structural else held_alpha
        cfg.training_arguments.adaptive_lambda.enabled = not structural
        OmegaConf.update(cfg, "channel_pruning", {"enabled": True, "mode": "explicit",
                         "structural": structural, "mask": mask}, merge=False, force_add=True)
        OmegaConf.update(cfg, "training_arguments.accuracy_guided_stage", {
            "id": spec["id"], "kind": spec["kind"], "end_global_epoch": end_epoch,
            "provenance": provenance, "held_alpha": held_alpha,
            "original_gate_normalization_metadata": normalization,
            "held_controller_state": deepcopy(controller_state) if structural else None}, force_add=True)
        _configure_run_history(cfg, output_root / spec["id"])
        cfg.mlflow.enabled = False
        cfg.mlflow.run_name = spec["id"]
        cfg.run_history.run_name = spec["id"]
        source_state = {name: tensor.detach().cpu().clone() for name, tensor in source.state_dict().items()}
        def initialize(model):
            model.load_state_dict(source_state, strict=True)
        def initialized(model, controller):
            if structural:
                if gates(model):
                    raise AssertionError("Physical recovery contains gates.")
                return
            validate_gate_normalization_metadata(model, normalization)
            if rebase is not None:
                before_alpha = held_alpha
                if not math.isclose(float(model.lambda_coef), before_alpha, rel_tol=1e-12):
                    raise AssertionError("Handoff reset selected alpha.")
                diagnostic = compare(accepted, model)
                report = check_transfer(model, accepted, mask)
                state["transitions"].append({**rebase, "alpha_preserved": before_alpha,
                    "first_forward_after_apply_initial_state": True,
                    "normalization": get_gate_regularization_diagnostics(model, before_alpha),
                    "predictors": diagnostic, "transfer_equivalence": report})
        def record(epoch, train, valid, model, optimizer, history):
            # The shared engine owns counters at each real optimizer update.
            event = deepcopy(getattr(history, "last_epoch_event", None))
            state["epoch_events"].append({"stage": spec["id"], "epoch": epoch,
                "ledger": deepcopy(ledger), "validation_accuracy": valid["valid_accuracy"],
                "actual_L_gate": float(train.get("train_L_gate_mean", train.get("train_regularization_loss", 0.0))),
                "controller_reason": "held_no_structural_gates" if structural else "accuracy_feedback",
                "alpha_base_held": held_alpha if structural else None,
                "checkpoint_event": event})
            persist()
        kwargs = dict(training_ledger=ledger, runtime_initialized_callback=initialized)
        if not structural:
            kwargs.update(adaptive_lambda_state=controller_state, adaptive_epoch_offset=offset,
                          adaptive_reference_by_epoch=reference, adaptive_rebase=rebase)
        result = run_training(cfg, model_initializer=initialize, epoch_end_callback=record, **kwargs)
        if result.get("test_metrics") or not result.get("test_evaluation_disabled"):
            raise RuntimeError("Unexpected test access in shared training engine.")
        if ledger["global_training_epoch"] != end_epoch or result["num_epochs_executed"] != epochs:
            raise RuntimeError("Incomplete stage; refusing an invented epoch budget.")
        checkpoint_dir = Path(result["run_dir"]) / "checkpoints"
        paths = sorted(checkpoint_dir.glob("epoch_*.pt"))
        if len(paths) != epochs:
            raise ValueError("V3 requires a complete atomic checkpoint event for every executed epoch.")
        records = []
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            evaluated = deepcopy(source).cpu()
            evaluated.load_state_dict(payload["model_state_dict"], strict=True)
            event = payload.get("epoch_event")
            if not isinstance(event, dict):
                raise ValueError("V3 selected checkpoint lacks its mandatory runtime event.")
            validate_epoch_eval_checkpoint(payload, evaluated)
            metrics = payload["metrics"]
            if structural:
                proposal, report = mask, {"params_after": sum(p.numel() for p in evaluated.parameters())}
            else:
                validate_gate_normalization_metadata(evaluated, normalization)
                proposal, report = select_learned_closed(evaluated, mask, float(c.eligibility.min_keep_ratio))
            records.append({"epoch": int(payload["epoch"]), "path": str(path),
                "accuracy": float(metrics["valid_accuracy"]), "ce_loss": float(metrics["valid_ce_loss"]),
                "physical_cost": report["params_after"], "proposal_mask": proposal, "selector": report})
        selected, selection = select_checkpoint_records(records, reference[end_epoch],
            float(config.training_arguments.adaptive_lambda.hard_drop), search=not structural)
        selection["reference_epoch"] = end_epoch
        selected_path = Path(selected["path"])
        payload = torch.load(selected_path, map_location="cpu", weights_only=True)
        model = deepcopy(source).cpu()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        validate_epoch_eval_checkpoint(payload, model)
        spec["selected_checkpoint"] = str(selected_path)
        summary = {**spec, "run_dir": result["run_dir"], "selection": selection,
                   "ledger_after_stage": deepcopy(ledger), "selected_model_hash": state_hash(model.state_dict()),
                   "checkpoint_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
                   "training_graph_cost": actual_training_graph_cost,
                   "training_graph_role": "physical_ungated" if structural else "full_width_masked_search_carrier",
                   "wall_seconds_including_stage_validation_checkpoint_io": time.perf_counter() - stage_started,
                   "memory": {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                              "scope": "process high-water mark, includes earlier stages and diagnostics",
                              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None},
                   "training_flops": None, "training_flops_status": "backward_not_measured; epochs_are_not_equal_compute"}
        state["stages"].append(summary)
        if not structural:
            controller_state = deepcopy(payload["epoch_event"]["controller_after_feedback"])
            held_alpha = float(payload["epoch_event"]["alpha_next"])
            summary.update(alpha_used=payload["epoch_event"]["alpha_used"], alpha_next=held_alpha)
        persist()
        return model, selected, selection

    persist()
    try:
        with cooperative_signals():
            for index in range(0, len(stage_plan), 3):
                search_spec, commit_spec, recovery_spec = stage_plan[index:index + 3]
                search, selected, selection = stage(search_spec, carrier, accepted_mask,
                                                     structural=False, rebase=pending_rebase)
                pending_rebase = None
                proposal_mask, selector_report = select_learned_closed(search, accepted_mask,
                                                                       float(c.eligibility.min_keep_ratio))
                if selection["no_feasible_search"]:
                    proposal_mask = deepcopy(accepted_mask)
                    proposed = deepcopy(selector_report)
                    selector_report = {**selector_report, "no_op_reason": "no_feasible_search",
                        "proposed_but_not_committed": proposed, "removed": [], "removed_original_ids": {},
                        "params_after": proposed["params_before"], "removed_params": 0,
                        "learned_candidates_materialized": 0, "compression_achieved": 0.0}
                old = build_structural(config, search, accepted_mask)
                candidate = build_structural(config, search, proposal_mask)
                if sum(p.numel() for p in old.parameters()) != selector_report["params_before"]:
                    raise AssertionError("Selector cost differs from the actual physical parent tensors.")
                if sum(p.numel() for p in candidate.parameters()) != selector_report["params_after"]:
                    raise AssertionError("Selector cost differs from the actually constructed export tensors.")
                # A technical slicing failure is immediate, before calibration or recovery.
                equivalence = check_transfer(search, candidate, proposal_mask)
                diagnostic_carrier = deepcopy(search)
                apply_channel_mask(diagnostic_carrier, proposal_mask)
                before_diagnostic = compare(candidate, diagnostic_carrier)
                blocked_closed = sum(int((~gate.get_hard_gate_decisions(apply_permanent_mask=False) &
                                          gate.get_permanent_survivor_mask().bool()).sum())
                                     for gate in gates(diagnostic_carrier).values())
                gated_equivalence = None
                if blocked_closed == 0:
                    # Here removal of gates should preserve the selected gated predictor.
                    gated_equivalence = gated_export_equivalence(diagnostic_carrier, candidate, sample, device)
                    state["orchestration_overhead"]["diagnostic_forward_examples"] += gated_equivalence["forward_examples"]
                    state["orchestration_overhead"]["diagnostic_wall_seconds"] += gated_equivalence["wall_seconds"]
                before = evaluate(candidate)
                calibration_accounting = {"batches": 0, "forward_examples": 0, "wall_seconds": 0.0}
                with isolated_diagnostic_rng(diagnostic_data.train_dataloader):
                    calibrate_bn(candidate, diagnostic_data.train_dataloader, device,
                        int(c.guard.train_bn_calibration_batches), seed + index, accounting=calibration_accounting)
                state["orchestration_overhead"]["calibration_forward_examples"] += calibration_accounting["forward_examples"]
                state["orchestration_overhead"]["calibration_wall_seconds"] += calibration_accounting["wall_seconds"]
                after_calibration = evaluate(candidate)
                if accepted is None:
                    accepted, accepted_metrics = old.cpu(), evaluate(old)
                    save_deployment({"stage": "first_search_all_open_physical_parent", "selected_checkpoint": selected["path"]})
                changed = proposal_mask != accepted_mask
                decision = {"stage": commit_spec["id"], "selected_checkpoint": selected["path"],
                    "selected_model_hash": state_hash(search.state_dict()),
                    "selected_eval_runtime": torch.load(selected["path"], map_location="cpu", weights_only=True)["epoch_event"]["eval_runtime"],
                    "selection": selection, "selector": selector_report, "proposal_mask": proposal_mask,
                    "accepted_mask_before": deepcopy(accepted_mask), "transfer_equivalence": equivalence,
                    "gated_export_equivalence": gated_equivalence,
                    "predictor_diagnostic_before_calibration": before_diagnostic,
                    "blocked_closed_survivors_opened_by_export": blocked_closed,
                    "opening_jump_expected": blocked_closed > 0,
                    "validation_before_calibration": before, "validation_after_calibration": after_calibration,
                    "validation_gated_before_surgery": {"accuracy": selected["accuracy"], "ce_loss": selected["ce_loss"]},
                    "calibration_overhead": calibration_accounting,
                    "physical_cost_before": measure_cost(old),
                    "physical_cost_after": measure_cost(candidate),
                    "new_pruning": changed}
                state["decisions"].append(decision)
                commit_spec["selected_checkpoint"] = selected["path"]
                # No parent-relative immediate accuracy guard: allocated recovery always runs.
                recovered, recovery_selected, _ = stage(recovery_spec, candidate, proposal_mask, structural=True)
                recovered_metrics = evaluate(recovered)
                threshold = reference[ledger["global_training_epoch"]] - float(config.training_arguments.adaptive_lambda.hard_drop)
                feasible = recovered_metrics["accuracy"] >= threshold
                decision.update(validation_after_recovery=recovered_metrics, quality_threshold=threshold,
                    reference_epoch=ledger["global_training_epoch"], quality_feasible=feasible,
                    status="accepted" if feasible else "rejected_after_recovery",
                    learned_candidates_materialized=len(selector_report["removed"]) if feasible else 0)
                if feasible:
                    previous_hash = mask_hash(accepted_mask)
                    accepted_mask, accepted, accepted_metrics = proposal_mask, recovered.cpu(), recovered_metrics
                    save_deployment({"stage": recovery_spec["id"], "selected_checkpoint": recovery_selected["path"]})
                    carrier = prepare_carry(accepted, search.cpu(), accepted_mask, normalization)
                    pending_rebase = {"transition_id": f"{recovery_spec['id']}->search_{index // 3 + 1}",
                        "phase_id": f"search_{index // 3 + 1}", "previous_mask_hash": previous_hash,
                        "new_mask_hash": mask_hash(accepted_mask), "reason": "physical_recovery_to_gated_search"}
                else:
                    remaining = total - ledger["global_training_epoch"]
                    decision.update(fallback_origin=deepcopy(accepted_origin), commits_stopped=True,
                                    rejected_recovery_epochs_charged=recovery_spec["epochs"])
                    if remaining:
                        fallback_spec = {"id": "fallback", "kind": "fallback", "epochs": remaining,
                            "commit_allowed": False, "selected_checkpoint": None, "restart_policy": "adamw_cosine_restart"}
                        fallback, fallback_selected, _ = stage(fallback_spec, accepted, accepted_mask, structural=True)
                        accepted, accepted_metrics = fallback.cpu(), evaluate(fallback)
                        save_deployment({"stage": "fallback", "selected_checkpoint": fallback_selected["path"]})
                    break
                persist()
        if ledger["global_training_epoch"] != total:
            raise AssertionError("Actual consumed epoch ledger does not match the allocated budget.")
        final_threshold = reference[total] - float(config.training_arguments.adaptive_lambda.hard_drop)
        final_cost = measure_cost(accepted, device=device)
        quality_feasible = accepted_metrics["accuracy"] >= final_threshold
        state.update(status="completed" if quality_feasible else "infeasible", validation=accepted_metrics,
                     quality_feasible=quality_feasible, final_quality_threshold=final_threshold,
                     final_cost=final_cost, compression_achieved=1 - final_cost["physical_total_parameters"] / initial_cost["physical_total_parameters"],
                     learned_candidates_materialized=sum(len(v) for v in accepted_mask.values()))
        # Persist final consumed ledger even when the selected model is an earlier checkpoint.
        save_deployment(None)
        persist()
        return state
    except Exception as exc:
        state.update(status="interrupted" if isinstance(exc, TrainingInterrupted) else "failed",
                     error=f"{type(exc).__name__}: {exc}", exact_whole_plan_resume_available=False)
        persist()
        raise
