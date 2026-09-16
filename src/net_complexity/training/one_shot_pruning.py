"""One shared adaptive search, one learned architecture, inherited/scratch finals.

The iterative algorithm is not invoked or changed. All optimization uses the
existing engine.  The inherited branch receives the selected search
checkpoint's sliced AdamW moments/steps; scratch starts with empty AdamW state.
Both physical branches deliberately restart their compact-stage cosine schedule.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from net_complexity.models.channel_pruning import build_structurally_pruned_model_from_config
from net_complexity.models.feature_selection import (
    get_gate_normalization_metadata, validate_gate_normalization_metadata,
)
from net_complexity.models.pruning_budget import gates, select_learned_closed
from .accuracy_guided_config import code_provenance, file_hash
from .accuracy_guided_pruning import atomic_checkpoint, select_checkpoint_records
from .cyclic_aig import _configure_run_history, _set_num_epochs
from .engine import _build_optimizer, run_training
from .interruption import cooperative_signals, TrainingInterrupted
from .one_shot_pruning_config import PROTOCOL, to_v3_config, validate_config, validate_inputs
from .one_shot_progress import epoch_progress, phase_progress, progress_message
from .optimizer_handoff import transfer_adamw_state_to_structural
from .pruning_audit import build_structural, committed_equivalence, load_adaptive_reference
from .pruning_measurement import (
    compare_predictors, deployment_cost, evaluate_deployment, gated_export_equivalence,
    isolated_diagnostic_rng, mask_hash, state_hash, write_json,
)
from .pruning_resume import validate_epoch_eval_checkpoint
from .randomness import set_random_seed


def physical_architecture_signature(model):
    """Structure and original-coordinate indices, independently of learned values."""
    shapes = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
              for name, tensor in model.state_dict().items()}
    indices = {name: tensor.cpu().tolist() for name, tensor in model.named_buffers()
               if not tensor.is_floating_point() and "num_batches_tracked" not in name}
    modules = {}
    attributes = ("in_channels", "out_channels", "kernel_size", "stride", "padding", "groups",
                  "in_features", "out_features", "num_features", "eps", "momentum", "affine")
    for name, module in model.named_modules():
        modules[name] = {"type": type(module).__module__ + "." + type(module).__qualname__,
                         **{key: getattr(module, key) for key in attributes if hasattr(module, key)}}
    return {"state_shapes": shapes, "topology_indices": indices, "modules": modules,
            "trainable": {name: parameter.requires_grad for name, parameter in model.named_parameters()}}


def build_physical_branches(config, selected_carrier, mask, selection_identity):
    """Transfer inherited state; construct scratch without any selected-state load."""
    cfg = to_v3_config(config)
    before = state_hash(selected_carrier.state_dict())
    if selection_identity.get("mask_hash") != mask_hash(mask):
        raise ValueError("Selection identity mask hash differs from the shared physical mask.")
    if selection_identity.get("selected_model_state_hash") != before:
        raise ValueError("Selection identity model hash differs from the selected checkpoint tensors.")
    with isolated_diagnostic_rng():
        inherited = build_structural(cfg, selected_carrier, mask).cpu()
    with isolated_diagnostic_rng():
        set_random_seed(int(config.seed))
        fresh_cfg = deepcopy(cfg)
        fresh_cfg.model.lambda_coef = 0.0
        pruning = OmegaConf.create({"enabled": True, "structural": True, "mode": "explicit", "mask": mask})
        # Fresh constructors initialize stem, classifier, every Conv/BN parameter
        # and every BN running buffer. There is intentionally no transfer/load.
        scratch = build_structurally_pruned_model_from_config(fresh_cfg, pruning).cpu()
    if state_hash(selected_carrier.state_dict()) != before:
        raise AssertionError("Constructing branches changed the selected checkpoint.")
    architecture = physical_architecture_signature(inherited)
    if architecture != physical_architecture_signature(scratch):
        raise AssertionError("Inherited/scratch physical architecture or original-coordinate indices differ.")
    if gates(inherited) or gates(scratch):
        raise AssertionError("Physical branches must contain no gates.")
    for module in scratch.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if (module.track_running_stats and (int(module.num_batches_tracked) != 0
                    or not torch.equal(module.running_mean, torch.zeros_like(module.running_mean))
                    or not torch.equal(module.running_var, torch.ones_like(module.running_var)))):
                raise AssertionError("Scratch inherited BN running statistics.")
            if module.affine and (not torch.equal(module.weight, torch.ones_like(module.weight))
                                  or not torch.equal(module.bias, torch.zeros_like(module.bias))):
                raise AssertionError("Scratch inherited BN affine state.")
    report = {**selection_identity,
        "architecture_hash": hashlib.sha256(json.dumps(architecture, sort_keys=True).encode()).hexdigest(),
        "inherited": {"initialization": "selected_surviving_state",
                      "initialization_state_hash": state_hash(inherited.state_dict())},
        "scratch": {"initialization": "pytorch_default_all_trainable_and_bn", "seed": int(config.seed),
                    "initialization_state_hash": state_hash(scratch.state_dict())}}
    return inherited, scratch, report


def _atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def run_one_shot_pruning(config, output_root):
    """Execute 60 shared search epochs and two independent 90-epoch final stages."""
    total = validate_config(config)
    with phase_progress("one-shot", "validating reference inputs"):
        inputs = validate_inputs(config)
    cfg = to_v3_config(config)
    search_epochs, final_epochs = int(config.one_shot.search_epochs), int(config.one_shot.final_epochs)
    reference = load_adaptive_reference(cfg.accuracy_guided.reference.history_path, total)
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError("Refusing overwrite/resume of an existing one-shot output directory.")
    device, seed = str(cfg.device), int(cfg.seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU full run.")
    with phase_progress("shared_search", "loading the zero-epoch initializer into the gated model"):
        set_random_seed(seed)
        carrier = instantiate(cfg.model)
        initial = torch.load(cfg.accuracy_guided.initializer.path, map_location="cpu", weights_only=True)
    if initial.get("trained_epochs") != 0 or initial.get("seed") != seed:
        raise ValueError("Search must start from the reference's matching zero-epoch initializer.")
    mismatch = carrier.load_state_dict(initial["model_state_dict"], strict=False)
    if mismatch.unexpected_keys or any("gumbel_layer" not in key for key in mismatch.missing_keys):
        raise ValueError(f"Incompatible zero-epoch backbone initializer: {mismatch}")
    del initial
    normalization = get_gate_normalization_metadata(carrier)
    with phase_progress("one-shot", "preparing validation loader and checking split"), isolated_diagnostic_rng():
        data = instantiate(cfg.dataloaders, include_test=False, loader_seed=seed)
        sample = next(iter(data.valid_dataloader))
    split_hash = mask_hash({name: list(getattr(loader.dataset, "indices", range(len(loader.dataset))))
        for name, loader in (("train", data.train_dataloader), ("valid", data.valid_dataloader))})
    reference_state = json.loads(Path(cfg.accuracy_guided.reference.state_path).read_text())
    if reference_state.get("split_indices_hash") != split_hash:
        raise ValueError("Actual train/validation split differs from the dense reference.")
    image_shape = tuple(sample[0].shape[1:])
    output_root.mkdir(parents=True)
    OmegaConf.save(config, output_root / "resolved_config.yaml", resolve=True)
    provenance = {"code": code_provenance(), "inputs": inputs, "seed": seed,
        "split_indices_hash": split_hash, "normalization": normalization,
        "resolved_config_sha256": hashlib.sha256(OmegaConf.to_yaml(config, resolve=True).encode()).hexdigest(),
        "torch_version": str(torch.__version__), "device": device,
        "source_weights": "shared_zero_epoch_initializer_only", "dense_reference_weights_loaded": False}
    search_ledger = dict(global_training_epoch=0, search_epochs_consumed=0,
                         optimizer_updates=0, consumed_training_examples=0)
    state = {"protocol": PROTOCOL, "status": "running", "test_evaluated": False,
        "provenance": provenance, "per_branch_total_allocated": total,
        "shared_search_ledger": search_ledger, "stages": {}, "selection": None,
        "export_only": None, "branches": {}, "compute_ledger": {}}
    started = time.perf_counter()

    def persist():
        finals = {name: max(0, record["ledger"]["global_training_epoch"] - search_epochs)
                  for name, record in state["branches"].items()}
        state["compute_ledger"] = {"shared_search_epochs": search_ledger["global_training_epoch"],
            "branch_final_training_epochs": finals,
            "actual_training_epochs_executed": search_ledger["global_training_epoch"] + sum(finals.values()),
            "actual_optimizer_updates": search_ledger["optimizer_updates"] + sum(
                record["ledger"]["optimizer_updates"] - search_ledger["optimizer_updates"]
                for record in state["branches"].values()),
            "actual_consumed_training_examples": search_ledger["consumed_training_examples"] + sum(
                record["ledger"]["consumed_training_examples"] - search_ledger["consumed_training_examples"]
                for record in state["branches"].values())}
        state["wall_seconds"] = time.perf_counter() - started
        write_json(output_root / "one_shot_state.json", state)

    def train_stage(
        name,
        source,
        mask,
        epochs,
        ledger,
        *,
        structural,
        identity=None,
        held_state=None,
        optimizer_initializer=None,
    ):
        stage_started = time.perf_counter()
        stage_cfg = deepcopy(cfg)
        _set_num_epochs(stage_cfg, epochs)
        offset = ledger["global_training_epoch"]
        stage_cfg.training_arguments.global_epoch_offset = offset
        stage_cfg.model.lambda_coef = 0.0 if structural else cfg.model.lambda_coef
        stage_cfg.training_arguments.adaptive_lambda.enabled = not structural
        OmegaConf.update(stage_cfg, "channel_pruning", {"enabled": True, "structural": structural,
            "mode": "explicit", "mask": mask}, merge=False, force_add=True)
        OmegaConf.update(stage_cfg, "training_arguments.accuracy_guided_stage", {
            "id": name, "kind": "physical_final" if structural else "search",
            "end_global_epoch": offset + epochs, "provenance": provenance,
            "selection_identity": identity, "held_controller_state": held_state,
            "original_gate_normalization_metadata": normalization}, force_add=True)
        stage_cfg.mlflow.run_name = stage_cfg.run_history.run_name = name
        _configure_run_history(stage_cfg, output_root / name / "training")
        source_state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
        initial_hash = state_hash(source_state)
        record = {"epochs_allocated": epochs, "global_epoch_offset": offset, "ledger": ledger,
            "optimizer": OmegaConf.to_container(stage_cfg.optimizer, resolve=True),
            "scheduler": OmegaConf.to_container(stage_cfg.scheduler, resolve=True),
            "optimizer_state_initialization": (
                "mapped_adamw_moments_and_step" if optimizer_initializer is not None else "fresh"
            ),
            "scheduler_state_initialization": "fresh",
            "initialization_state_hash": initial_hash, "training_initializer_verified": False,
            "epoch_events": []}
        state["stages"][name] = record

        def initialize(model):
            model.load_state_dict(source_state, strict=True)

        def initialized(model, controller):
            if state_hash(model.state_dict()) != initial_hash:
                raise AssertionError("Final apply_initial_state changed the intended branch weights/BN state.")
            if structural and (gates(model) or model.lambda_coef != 0 or controller is not None):
                raise AssertionError("Physical final training has gates, gate penalty, or live controller.")
            if not structural:
                validate_gate_normalization_metadata(model, normalization)
            record["training_initializer_verified"] = True

        def initialize_optimizer(model, optimizer):
            handoff = optimizer_initializer(model, optimizer)
            record["optimizer_handoff"] = deepcopy(handoff)
            return handoff

        def epoch_end(epoch, train, valid, model, optimizer, history):
            record["epoch_events"].append({"epoch": epoch, "ledger": deepcopy(ledger),
                "validation": {"accuracy": valid["valid_accuracy"], "ce_loss": valid["valid_ce_loss"]},
                "train_L_gate_mean": train["train_L_gate_mean"]})
            persist()
            epoch_progress(name, epoch, epochs, valid, ledger, time.perf_counter() - stage_started)

        kwargs = {"training_ledger": ledger, "runtime_initialized_callback": initialized}
        if optimizer_initializer is not None:
            kwargs["optimizer_initializer"] = initialize_optimizer
        if not structural:
            kwargs.update(adaptive_epoch_offset=0, adaptive_reference_by_epoch=reference)
        with phase_progress(name, f"training {epochs} epochs on {device}; state={output_root / 'one_shot_state.json'}",
                            ledger=ledger, total_epochs=epochs):
            result = run_training(stage_cfg, model_initializer=initialize, epoch_end_callback=epoch_end, **kwargs)
        if result.get("test_metrics") or not result.get("test_evaluation_disabled"):
            raise AssertionError("One-shot training accessed test data.")
        if result["num_epochs_executed"] != epochs or ledger["global_training_epoch"] != offset + epochs:
            raise RuntimeError("Incomplete training stage; consumed budget cannot be fabricated.")
        record.update(run_dir=result["run_dir"], status="completed")
        paths = sorted((Path(result["run_dir"]) / "checkpoints").glob("epoch_*.pt"))
        if len(paths) != epochs:
            raise ValueError("Missing complete epoch candidates for validation-only selection.")
        return paths, record

    def load_model(path, source):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = deepcopy(source).cpu()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        validate_epoch_eval_checkpoint(checkpoint, model)
        return model, checkpoint

    persist()
    try:
        with cooperative_signals():
            paths, _ = train_stage("shared_search", carrier, {}, search_epochs, search_ledger, structural=False)
            candidates = []
            with phase_progress("selection", f"checking {len(paths)} search checkpoints by validation and physical size"):
                for index, path in enumerate(paths, 1):
                    model, checkpoint = load_model(path, carrier)
                    mask, selector = select_learned_closed(model, {}, float(cfg.accuracy_guided.eligibility.min_keep_ratio))
                    candidates.append({"epoch": int(checkpoint["epoch"]), "path": str(path),
                        "accuracy": float(checkpoint["metrics"]["valid_accuracy"]),
                        "ce_loss": float(checkpoint["metrics"]["valid_ce_loss"]),
                        "physical_cost": selector["params_after"], "pruning_mask": mask, "selector": selector})
                    if index % 10 == 0 or index == len(paths):
                        progress_message("selection", f"checked {index}/{len(paths)} checkpoints")
            selected, selection_trace = select_checkpoint_records(candidates, reference[search_epochs],
                float(cfg.training_arguments.adaptive_lambda.hard_drop), search=True)
            selection_trace["reference_epoch"] = search_epochs
            if selection_trace["no_feasible_search"]:
                state.update(status="no_feasible_search", selection=selection_trace)
                write_json(output_root / "selection.json", selection_trace)
                persist()
                progress_message("selection", "no feasible search checkpoint; stopping before export and branch training")
                return state
            selected_carrier, checkpoint = load_model(selected["path"], carrier)
            if "optimizer_state_dict" not in checkpoint:
                raise ValueError("Selected search checkpoint has no optimizer_state_dict for handoff.")
            selected_optimizer, _ = _build_optimizer(cfg, selected_carrier)
            selected_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            mask = selected["pruning_mask"]
            selected_hash = file_hash(selected["path"])
            identity = {"selected_checkpoint_id": f"shared_search:epoch_{selected['epoch']:04d}:{selected_hash[:16]}",
                "selected_checkpoint_hash": selected_hash,
                "selected_model_state_hash": state_hash(selected_carrier.state_dict()), "mask_hash": mask_hash(mask)}
            selection = {**identity, "pruning_mask": mask, "selected_epoch": selected["epoch"],
                "quality_threshold": selection_trace["quality_threshold"], "reference_epoch": search_epochs,
                "policy": "best_feasible_compact", "trace": selection_trace,
                "search_ledger_consumed": deepcopy(search_ledger)}
            _atomic_copy(selected["path"], output_root / "selected_checkpoint.pt")
            state["selection"] = selection
            write_json(output_root / "selection.json", selection)
            progress_message("selection", f"selected {identity['selected_checkpoint_id']}; physical_parameters={selected['physical_cost']}")
            inherited, scratch, initialization = build_physical_branches(config, selected_carrier, mask, identity)
            if sum(p.numel() for p in inherited.parameters()) != selected["physical_cost"]:
                raise AssertionError("Selected cost estimate differs from the actual compact tensors.")
            export_dir = output_root / "export_only"
            export_dir.mkdir()
            source_hash = state_hash(selected_carrier.state_dict())
            physical_hash = state_hash(inherited.state_dict())
            # This compares the actual selected gated predictor, with its original
            # runtime/mask, to the exported model. The transfer check is separate.
            with phase_progress("export_only", "frozen validation comparison before training or BN recalibration"):
                comparison = compare_predictors(inherited, selected_carrier, data.valid_dataloader, device)
            report = {**identity, "pruning_mask": mask, "architecture_hash": initialization["architecture_hash"],
                "status": "measured", "training_epochs": 0, "bn_calibration_batches": 0,
                "gated_validation": comparison["carry_carrier"], "physical_validation": comparison["physical"],
                "logits_comparison": comparison["carry_vs_physical"], "diagnostic_overhead": comparison["overhead"],
                "selector": selected["selector"], "physical_state_hash": physical_hash,
                "physical_cost": deployment_cost(inherited, image_shape=image_shape),
                "search_carrier_cost": deployment_cost(selected_carrier, image_shape=image_shape),
                "search_carrier_scope": "full-width gated carrier, not a compact model"}
            state["export_only"] = report
            write_json(export_dir / "diagnostics.json", report)
            try:
                with phase_progress("export_only", "checking physical Conv/BN transfer equivalence"):
                    report["transfer_equivalence"] = committed_equivalence(selected_carrier, inherited, mask, sample,
                        device, report_path=export_dir / "transfer_equivalence.json")
            except Exception as exc:
                report.update(status="technical_transfer_failure", non_equivalence_reason=str(exc))
                write_json(export_dir / "diagnostics.json", report)
                raise
            disabled = {name: set(ids) for name, ids in mask.items()}
            blocked_closed = [{"boundary": name, "original_id": i}
                for name, gate in gates(selected_carrier).items()
                for i, opened in enumerate(gate.get_hard_gate_decisions(apply_permanent_mask=False).tolist())
                if not opened and i not in disabled.get(name, set())]
            report["blocked_closed_survivors_opened_by_export"] = blocked_closed
            try:
                report["gated_equivalence"] = gated_export_equivalence(selected_carrier, inherited, sample, device)
                report["gated_equivalent_on_checked_batch"] = True
            except AssertionError as exc:
                report.update(gated_equivalent_on_checked_batch=False,
                    non_equivalence_reason="blocked_closed_survivors_opened" if blocked_closed else "gated_export_function_difference",
                    gated_equivalence_error=str(exc))
            if source_hash != state_hash(selected_carrier.state_dict()) or physical_hash != state_hash(inherited.state_dict()):
                raise AssertionError("Export diagnostics modified weights or BN state.")
            report["weights_and_bn_unchanged"] = True
            if not report["gated_equivalent_on_checked_batch"] and not blocked_closed:
                report["status"] = "technical_gated_export_failure"
            write_json(export_dir / "diagnostics.json", report)
            if report["status"] == "technical_gated_export_failure":
                raise RuntimeError("Unexplained gated/physical export mismatch; see export_only/diagnostics.json")
            atomic_checkpoint(export_dir / "deployment.pt", {**identity, "protocol": PROTOCOL,
                "artifact_type": "physical_ungated", "model_state_dict": inherited.state_dict(),
                "model_state_hash": physical_hash, "pruning_mask": mask, "training_epochs": 0,
                "bn_calibration_batches": 0, "normalization_metadata": normalization})
            persist()
            progress_message("export_only", f"diagnostics saved: {export_dir / 'diagnostics.json'}")
            held_controller = checkpoint["epoch_event"]["controller_after_feedback"]
            for name, model in (("inherited", inherited), ("scratch", scratch)):
                branch_dir = output_root / name
                branch_dir.mkdir()
                ledger = deepcopy(search_ledger)
                branch = {**identity, "protocol": PROTOCOL, "artifact_type": "physical_ungated", "branch": name,
                    "status": "running", "pruning_mask": mask, "architecture_hash": initialization["architecture_hash"],
                    "initialization_state_hash": initialization[name]["initialization_state_hash"],
                    "initialization": initialization[name]["initialization"], "ledger": ledger,
                    "optimizer_state_initialization": (
                        "mapped_adamw_moments_and_step" if name == "inherited" else "fresh"
                    ),
                    "scheduler_state_initialization": "fresh",
                    "normalization_metadata": normalization, "provenance": provenance, "test_evaluated": False}
                state["branches"][name] = branch
                atomic_checkpoint(branch_dir / "initial_state.pt", {**branch,
                    "model_state_dict": model.cpu().state_dict(), "model_state_hash": state_hash(model.state_dict())})
                optimizer_initializer = None
                if name == "inherited":
                    def optimizer_initializer(target_model, target_optimizer):
                        return transfer_adamw_state_to_structural(
                            selected_carrier,
                            selected_optimizer,
                            target_model,
                            target_optimizer,
                        )
                paths, stage_record = train_stage(
                    name,
                    model,
                    mask,
                    final_epochs,
                    ledger,
                    structural=True,
                    identity=identity,
                    held_state=held_controller,
                    optimizer_initializer=optimizer_initializer,
                )
                final_candidates = []
                with phase_progress(name, f"selecting the frozen deployment from {len(paths)} validation checkpoints"):
                    for path in paths:
                        payload = torch.load(path, map_location="cpu", weights_only=True)
                        final_candidates.append({"path": str(path), "epoch": int(payload["epoch"]),
                            "accuracy": float(payload["metrics"]["valid_accuracy"]),
                            "ce_loss": float(payload["metrics"]["valid_ce_loss"]), "physical_cost": selected["physical_cost"]})
                final_selected, final_selection = select_checkpoint_records(final_candidates, reference[total],
                    float(cfg.training_arguments.adaptive_lambda.hard_drop), search=False)
                deployed, _ = load_model(final_selected["path"], model)
                before = state_hash(deployed.state_dict())
                with phase_progress(name, "validating selected frozen deployment"), isolated_diagnostic_rng(data.valid_dataloader):
                    validation = evaluate_deployment(deployed, data.valid_dataloader, device)
                if state_hash(deployed.state_dict()) != before:
                    raise AssertionError("Frozen deployment validation changed state.")
                deployed.cpu()
                feasible = validation["accuracy"] >= final_selection["quality_threshold"]
                branch.update(status="completed" if feasible else "infeasible", validation=validation,
                    quality_feasible=feasible, quality_threshold=final_selection["quality_threshold"],
                    final_cost=deployment_cost(deployed, image_shape=image_shape), model_state_hash=before,
                    selected_final_checkpoint=str(final_selected["path"]), selected_final_epoch=final_selected["epoch"],
                    training_initializer_verified=stage_record["training_initializer_verified"],
                    optimizer_handoff=stage_record.get("optimizer_handoff"),
                    final_training_epochs_executed=final_epochs, per_branch_total_allocated=total,
                    selection_policy=final_selection["policy"], reference_epoch=total)
                atomic_checkpoint(branch_dir / "deployment.pt", {**branch, "model_state_dict": deployed.state_dict()})
                write_json(branch_dir / "branch_state.json", branch)
                persist()
                progress_message(name, f"{branch['status']}; validation={validation['accuracy']:.4%}; deployment={branch_dir / 'deployment.pt'}")
            if any(branch["ledger"]["global_training_epoch"] != total for branch in state["branches"].values()):
                raise AssertionError("Each branch must consume its complete attributed search+final budget.")
            state["status"] = "completed"
            write_json(output_root / "comparison.json", {"protocol": PROTOCOL, **identity,
                "architecture_hash": initialization["architecture_hash"],
                "validation": {name: branch["validation"] for name, branch in state["branches"].items()},
                "quality_feasible": {name: branch["quality_feasible"] for name, branch in state["branches"].items()},
                "official_test": None, "test_evaluated": False})
            persist()
            return state
    except Exception as exc:
        state.update(status="interrupted" if isinstance(exc, TrainingInterrupted) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        persist()
        progress_message("one-shot", f"{state['status']}: {state['error']}; state={output_root / 'one_shot_state.json'}")
        raise
