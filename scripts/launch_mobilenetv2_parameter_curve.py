"""Run the checked four-point MobileNetV2 adaptive-pruning curve overnight."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from net_complexity.models.pruning_budget import PhysicalBudget, gates
from net_complexity.training.one_shot_pruning_config import (
    compose_config,
    resolved_one_shot,
    to_v3_config,
    validate_config,
    validate_inputs,
)
from net_complexity.training.pruning_audit import build_structural


ONE_SHOT = ROOT / "scripts/launch_one_shot_pruning.py"
DEFAULT_PLAN = "mobilenetv2_parameter_curve_60_90_nightly"
PLAN_FIELDS = {
    "name", "search_config", "recovery_config", "reference_output", "output",
    "data", "test_config", "run_official_test", "dense_physical_parameters",
    "minimum_physical_parameters", "points",
}
POINT_FIELDS = {
    "id", "soft_drop", "hard_drop", "target_hint_parameters",
    "resnet50_analogue_parameters",
}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _root_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_plan(config_name=DEFAULT_PLAN):
    name = str(config_name).removesuffix(".yaml")
    if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        raise ValueError("--config-name must name one root YAML under configs")
    path = ROOT / "configs" / f"{name}.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or set(raw) != PLAN_FIELDS:
        raise ValueError(
            f"Nightly plan fields differ: unknown={sorted(set(raw or {}) - PLAN_FIELDS)}, "
            f"missing={sorted(PLAN_FIELDS - set(raw or {}))}"
        )
    if raw["name"] != name:
        raise ValueError("Plan name must equal its YAML filename")
    if not isinstance(raw["run_official_test"], bool):
        raise ValueError("run_official_test must be boolean")
    for key in ("dense_physical_parameters", "minimum_physical_parameters"):
        if type(raw[key]) is not int or raw[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if raw["minimum_physical_parameters"] >= raw["dense_physical_parameters"]:
        raise ValueError("minimum physical size must be smaller than dense physical size")
    points = raw["points"]
    if not isinstance(points, list) or len(points) != 4:
        raise ValueError("The checked curve requires exactly four points")
    ids = []
    previous_target = math.inf
    previous_soft = -math.inf
    previous_hard = -math.inf
    for point in points:
        if not isinstance(point, dict) or set(point) != POINT_FIELDS:
            raise ValueError("Every point must contain exactly the checked curve fields")
        if not isinstance(point["id"], str) or re.fullmatch(r"[a-z0-9_]+", point["id"]) is None:
            raise ValueError("Point ids must be safe lowercase path names")
        ids.append(point["id"])
        soft, hard = point["soft_drop"], point["hard_drop"]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in (soft, hard)):
            raise ValueError("Drop thresholds must be finite numbers")
        if not 0 <= soft < hard <= 1 or soft < previous_soft or hard < previous_hard:
            raise ValueError("Points must progress through nondecreasing valid drop bands")
        target = point["target_hint_parameters"]
        if type(target) is not int or not raw["minimum_physical_parameters"] <= target <= raw["dense_physical_parameters"]:
            raise ValueError("Every MobileNet target hint must be within its physical range")
        if target >= previous_target:
            raise ValueError("Target hints must decrease as the drop band becomes more permissive")
        if type(point["resnet50_analogue_parameters"]) is not int:
            raise ValueError("ResNet50 analogue sizes must be integers")
        previous_soft, previous_hard, previous_target = soft, hard, target
    if len(set(ids)) != len(ids):
        raise ValueError("Point ids must be unique")
    return raw


def _overrides(point):
    return [
        f"training_arguments.adaptive_lambda.soft_drop={point['soft_drop']}",
        f"training_arguments.adaptive_lambda.hard_drop={point['hard_drop']}",
    ]


def _compose_profiles(plan, dense_source=None):
    profiles = []
    for point in plan["points"]:
        overrides = _overrides(point)
        search = compose_config(plan["search_config"], overrides, dense_source=dense_source)
        recovery = compose_config(plan["recovery_config"], overrides, dense_source=dense_source)
        if validate_config(search) != 150 or validate_config(recovery) != 150:
            raise AssertionError("Every curve point must retain the 150-epoch model budget")
        if (int(search.one_shot.search_epochs), int(search.one_shot.final_epochs)) != (60, 90):
            raise AssertionError("Search profile is not exactly 60/90")
        if (int(recovery.one_shot.search_epochs), int(recovery.one_shot.final_epochs)) != (60, 90):
            raise AssertionError("Recovery profile is not exactly 60/90")
        if not bool(search.training_arguments.adaptive_lambda.enabled):
            raise AssertionError("Adaptive lambda is disabled")
        if len(resolved_one_shot(recovery, check_inputs=False)["execution_policy"]["branch_plan"]) != 1:
            raise AssertionError("Each point must have exactly one physical recovery")
        profiles.append((point, search, recovery))
    return profiles


def _validate_dataset(data_root):
    root = Path(data_root)
    wnids = root / "wnids.txt"
    annotations = root / "val/val_annotations.txt"
    val_images = root / "val/images"
    train = root / "train"
    missing = [str(path) for path in (wnids, annotations, val_images, train) if not path.exists()]
    if missing:
        raise FileNotFoundError("TinyImageNet-200 is incomplete; missing: " + ", ".join(missing))
    classes = [line.strip() for line in wnids.read_text().splitlines() if line.strip()]
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError("TinyImageNet-200 wnids.txt must contain 200 unique classes")
    if sum(1 for line in annotations.read_text().splitlines() if line.strip()) != 10_000:
        raise ValueError("TinyImageNet-200 official validation annotations must contain 10,000 rows")
    absent_train = [name for name in classes if not (train / name / "images").is_dir()]
    if absent_train:
        raise FileNotFoundError(f"TinyImageNet-200 train split is missing {len(absent_train)} class image directories")


def _validate_test_config(path, data_root):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Official-test config does not exist: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    values = raw.get("one_shot_test") if isinstance(raw, dict) else None
    if not isinstance(values, dict):
        raise ValueError("Official-test config must contain one_shot_test")
    configured_data = _root_path(values.get("data", ""))
    if configured_data != Path(data_root).resolve():
        raise ValueError(
            f"Training and official-test data roots differ: {data_root} != {configured_data}"
        )
    if values.get("download") is not False:
        raise ValueError("TinyImageNet official-test config must never download implicitly")


def _model_preflight(plan, search_config, *, require_cuda):
    model = instantiate(search_config.model).cpu()
    physical_dense = PhysicalBudget(model, {}).total()
    if physical_dense != int(plan["dense_physical_parameters"]):
        raise AssertionError(
            f"Dense MobileNetV2 parameter contract changed: {physical_dense} != "
            f"{plan['dense_physical_parameters']}"
        )
    ratio = float(search_config.accuracy_guided.eligibility.min_keep_ratio)
    mask = {}
    for name, gate in gates(model).items():
        keep = max(1, math.ceil(gate.initial_channels * ratio))
        mask[name] = list(range(keep, gate.initial_channels))
    floor = PhysicalBudget(model, mask).total()
    if floor != int(plan["minimum_physical_parameters"]):
        raise AssertionError(
            f"MobileNetV2 structural floor changed: {floor} != "
            f"{plan['minimum_physical_parameters']}"
        )
    physical = build_structural(to_v3_config(search_config), model, mask).cpu()
    if sum(parameter.numel() for parameter in physical.parameters()) != floor:
        raise AssertionError("Physical tensors differ from the exact MobileNetV2 budget")
    if gates(physical):
        raise AssertionError("Physical recovery model still contains gates")
    if not require_cuda:
        return {"dense_physical_parameters": physical_dense, "minimum_physical_parameters": floor}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing an overnight CPU launch")
    device = torch.device("cuda:0")
    probe = instantiate(search_config.model).to(device).train()
    x = torch.randn(2, 3, 64, 64, device=device)
    y = torch.tensor([0, 1], device=device)
    output = probe(x, y)
    if not torch.isfinite(output.loss):
        raise FloatingPointError("Non-finite MobileNetV2 preflight loss")
    output.loss.backward()
    gate_gradients = [parameter.grad for name, parameter in probe.named_parameters() if "logits" in name]
    if not gate_gradients or any(gradient is None or not torch.isfinite(gradient).all() for gradient in gate_gradients):
        raise RuntimeError("Adaptive gate gradients are missing or non-finite on CUDA")
    return {
        "dense_physical_parameters": physical_dense,
        "minimum_physical_parameters": floor,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


def _command(config_name, dense_source, output, point, *, search_only=False, reuse_search=None,
             test_config=None, skip_test=False):
    command = [
        sys.executable, str(ONE_SHOT), "--config-name", config_name,
        "--dense-source", str(dense_source), "--output", str(output),
    ]
    for override in _overrides(point):
        command.extend(["--override", override])
    if search_only:
        command.append("--search-only")
    if reuse_search is not None:
        command.extend(["--reuse-search", str(reuse_search)])
    if test_config is not None:
        command.extend(["--test-config", str(test_config)])
    if skip_test:
        command.append("--skip-test")
    return command


def _run(command):
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Child command failed with exit code {result.returncode}: "
            + " ".join(map(str, command))
        )


def _search_record(run_dir):
    state = json.loads((run_dir / "one_shot_state.json").read_text())
    diagnostics = json.loads((run_dir / "export_only/diagnostics.json").read_text())
    selection = json.loads((run_dir / "selection.json").read_text())
    if state.get("status") != "search_only_completed":
        raise RuntimeError(f"Search did not complete: status={state.get('status')}")
    if int(state["shared_search_ledger"]["search_epochs_consumed"]) != 60:
        raise RuntimeError("Search did not consume exactly 60 epochs")
    return {
        "selected_epoch": int(selection["selected_epoch"]),
        "selection_policy": selection["policy"],
        "physical_parameters": int(diagnostics["physical_cost"]["physical_total_parameters"]),
        "physical_validation_accuracy": float(diagnostics["physical_validation"]["accuracy"]),
        "gated_equivalent_on_checked_batch": bool(diagnostics["gated_equivalent_on_checked_batch"]),
    }


def _recovery_record(run_dir):
    state = json.loads((run_dir / "one_shot_state.json").read_text())
    if state.get("status") != "completed":
        raise RuntimeError(f"Recovery did not complete: status={state.get('status')}")
    if len(state.get("branches", {})) != 1:
        raise RuntimeError("Recovery produced something other than one physical branch")
    branch_name, branch = next(iter(state["branches"].items()))
    if int(branch["ledger"]["global_training_epoch"]) != 150:
        raise RuntimeError("Recovered deployment does not have a complete 60+90 ledger")
    result = {
        "branch": branch_name,
        "status": branch["status"],
        "validation_accuracy": float(branch["validation"]["accuracy"]),
        "physical_parameters": int(branch["final_cost"]["physical_total_parameters"]),
        "quality_feasible": bool(branch["quality_feasible"]),
    }
    test_path = run_dir / "test_evaluation/test_summary.json"
    if test_path.is_file():
        test = json.loads(test_path.read_text())
        if test.get("status") != "completed" or len(test.get("runs", [])) != 1:
            raise RuntimeError("Official frozen test evaluation is incomplete")
        result["official_test_accuracy"] = float(test["runs"][0]["test"]["accuracy"])
        result["official_test_summary"] = str(test_path)
    return result


def _preview(plan, dense_source, output, reference_output, *, from_scratch):
    profiles = _compose_profiles(plan, dense_source=None)
    points = []
    for point, search, recovery in profiles:
        points.append({
            **deepcopy(point),
            "search_epochs": int(search.one_shot.search_epochs),
            "recovery_epochs": int(recovery.one_shot.final_epochs),
            "physical_recovery_branches": 1,
            "adaptive_lambda_enabled": bool(search.training_arguments.adaptive_lambda.enabled),
            "search_output": str(output / point["id"] / "search60"),
            "recovery_output": str(output / point["id"] / "recovery90"),
        })
    return {
        "name": plan["name"], "mode": "from_scratch" if from_scratch else "reuse_dense_reference",
        "training_performed": False,
        "dense_source": str(dense_source) if dense_source else None,
        "new_dense_reference": str(reference_output) if from_scratch else None,
        "output": str(output), "points": points,
        "budget": {
            "dense_reference_epochs": 150 if from_scratch else 0,
            "epochs_per_curve_model": 150,
            "curve_models": 4,
            "total_curve_training_epochs": 600,
            "total_training_epochs_this_launch": 750 if from_scratch else 600,
        },
        "official_test_after_each_frozen_deployment": bool(plan["run_official_test"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_PLAN)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dense-source", type=Path, help="Existing unpacked MobileNetV2 dense-reference run")
    source.add_argument("--from-scratch", action="store_true", help="Train one new dense150 reference first")
    parser.add_argument("--reference-output", type=Path, help="Fresh dense-reference output for --from-scratch")
    parser.add_argument("--output", type=Path, help="Fresh parent directory for all four curve points")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact plan without CUDA, data, writes, or training")
    parser.add_argument("--preflight-only", action="store_true", help="Check data, CUDA, model/export and reference inputs, then stop")
    args = parser.parse_args(argv)
    plan = load_plan(args.config_name)
    output = (args.output.expanduser().resolve() if args.output else _root_path(plan["output"]))
    reference_output = (
        args.reference_output.expanduser().resolve()
        if args.reference_output else _root_path(plan["reference_output"])
    )
    if args.reference_output is not None and not args.from_scratch:
        parser.error("--reference-output requires --from-scratch")
    dense_source = args.dense_source.expanduser().resolve() if args.dense_source else None
    if dense_source is None and not args.from_scratch and reference_output.exists():
        dense_source = reference_output
    if dense_source is None and not args.from_scratch:
        parser.error("Pass --dense-source PATH or --from-scratch; no default dense reference exists")
    if args.dry_run:
        report = _preview(plan, dense_source, output, reference_output, from_scratch=args.from_scratch)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    if sys.version_info < (3, 10):
        raise RuntimeError("Server runtime requires Python >=3.10")

    data = _root_path(plan["data"])
    test_config = _root_path(plan["test_config"])
    _validate_dataset(data)
    _validate_test_config(test_config, data)
    profiles = _compose_profiles(plan, dense_source=dense_source)
    preflight = _model_preflight(plan, profiles[0][1], require_cuda=True)
    if dense_source is not None:
        for _, search, recovery in profiles:
            validate_inputs(search)
            validate_inputs(recovery)
    if args.from_scratch and reference_output.exists():
        raise FileExistsError(f"Refusing existing dense reference output: {reference_output}")
    if not args.from_scratch and (dense_source is None or not dense_source.is_dir()):
        raise FileNotFoundError(f"Dense reference directory does not exist: {dense_source}")
    if output.exists():
        raise FileExistsError(f"Refusing existing curve output: {output}")
    free = shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free
    if free < 20 * 1024 ** 3:
        raise OSError("At least 20 GiB free disk is required for four checkpoint histories")
    if args.preflight_only:
        report = {"status": "preflight_passed", "training_performed": False, **preflight}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report

    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "nightly_state.json"
    state = {
        "protocol": "mobilenetv2_parameter_curve_60_90_v1",
        "status": "running", "started_unix": time(),
        "plan": deepcopy(plan), "preflight": preflight,
        "dense_source": str(dense_source) if dense_source else None,
        "reference": None, "points": {}, "test_evaluated": False,
    }
    _write_json(state_path, state)
    try:
        if args.from_scratch:
            print(f"[nightly] dense reference: {reference_output}", flush=True)
            _run([
                sys.executable, str(ONE_SHOT), "--config-name", plan["search_config"],
                "--prepare-reference", str(reference_output),
            ])
            dense_source = reference_output
            state["dense_source"] = str(dense_source)
            reference_state = json.loads((reference_output / "J1_dense_control/pilot_state.json").read_text())
            if reference_state.get("status") != "completed" or int(reference_state["global_epochs_completed"]) != 150:
                raise RuntimeError("New dense reference did not complete exactly 150 epochs")
            state["reference"] = {
                "status": "completed", "path": str(reference_output),
                "training_epochs": 150,
            }
            # Validate the actual freshly written immutable inputs before search.
            profiles = _compose_profiles(plan, dense_source=dense_source)
            for _, search, recovery in profiles:
                validate_inputs(search)
                validate_inputs(recovery)
            _write_json(state_path, state)
        else:
            state["reference"] = {"status": "reused", "path": str(dense_source), "training_epochs": 0}
            _write_json(state_path, state)

        for index, (point, _, _) in enumerate(profiles, 1):
            point_root = output / point["id"]
            search_dir, recovery_dir = point_root / "search60", point_root / "recovery90"
            print(
                f"[nightly] point {index}/4 {point['id']}: "
                f"soft={point['soft_drop']:.4f} hard={point['hard_drop']:.4f}",
                flush=True,
            )
            record = {**deepcopy(point), "status": "searching", "search": None, "recovery": None}
            state["points"][point["id"]] = record
            _write_json(state_path, state)
            _run(_command(
                plan["search_config"], dense_source, search_dir, point,
                search_only=True, test_config=test_config,
            ))
            record["search"] = _search_record(search_dir)
            record["status"] = "recovering"
            _write_json(state_path, state)
            _run(_command(
                plan["recovery_config"], dense_source, recovery_dir, point,
                reuse_search=search_dir, test_config=test_config,
                skip_test=not plan["run_official_test"],
            ))
            record["recovery"] = _recovery_record(recovery_dir)
            if record["recovery"]["physical_parameters"] != record["search"]["physical_parameters"]:
                raise AssertionError("Recovery architecture size differs from its selected search export")
            record["target_hint_error_parameters"] = (
                record["recovery"]["physical_parameters"] - point["target_hint_parameters"]
            )
            record["status"] = "completed"
            state["test_evaluated"] = state["test_evaluated"] or (
                "official_test_accuracy" in record["recovery"]
            )
            _write_json(state_path, state)

        state["status"] = "completed"
        state["finished_unix"] = time()
        _write_json(state_path, state)
        _write_json(output / "nightly_summary.json", state)
        print(f"[nightly] completed: {output / 'nightly_summary.json'}", flush=True)
        return state
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["finished_unix"] = time()
        _write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
