"""Run exactly four expected-open-count ResNet50/CIFAR-10 models.

The checked-in suite is the Cartesian product of two accepted validation-drop
points and entropy beta in {0.0, 0.3}. Every model performs one 60-epoch
adaptive-lambda search followed by one 90-epoch physical recovery. There are no
scratch branches, repeats, curve points, or other hidden training jobs. Epoch
checkpoints exist only while selecting one model; after its search and recovery
are verified, nested checkpoints are removed and root selected/deployment
artifacts are retained.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
ONE_SHOT = ROOT / "scripts/launch_one_shot_pruning.py"
SUITE_CONFIG = (
    ROOT
    / "configs/experiment/pruning_v3/resnet50_cifar10_expected_open_four_runs.yaml"
)
DEFAULT_DENSE_SOURCE = Path("outputs/baseline/one_shot_dense_reference_seed42")
DEFAULT_OUTPUT = Path(
    "outputs/runs/resnet50_cifar10_expected_open_four_runs_online_selection_v1"
)
EXPECTED_POINTS = {
    (0.0005, 0.001),
    (0.01, 0.02),
}
EXPECTED_BETAS = {0.0, 0.3}
DEFAULT_MIN_FREE_GIB = 10.0


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _plain_suite(path=SUITE_CONFIG):
    suite = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    _require(isinstance(suite, dict), "Suite config must be a mapping")
    _require(
        set(suite) == {
            "protocol", "search_config", "recovery_config", "drop_mode",
            "search_epochs", "recovery_epochs", "runs",
        },
        "Suite config fields changed",
    )
    return suite


def _overrides(row):
    return (
        f"training_arguments.adaptive_lambda.soft_drop={row['soft_drop']:.12g}",
        f"training_arguments.adaptive_lambda.hard_drop={row['hard_drop']:.12g}",
        "model.entropy_regularization=plus_negative_entropy",
        f"model.entropy_regularization_coef={row['entropy_beta']:.12g}",
    )


def validate_suite_configs(path=SUITE_CONFIG):
    """Resolve the suite and prove that it contains exactly four 150-epoch models."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from net_complexity.training import one_shot_pruning_config as schema

    suite = _plain_suite(path)
    _require(
        suite["protocol"] == "resnet50_cifar10_expected_open_four_runs_v1",
        "Unexpected suite protocol",
    )
    _require(suite["drop_mode"] == "expected_open_count", "Unexpected selector")
    _require(
        (suite["search_epochs"], suite["recovery_epochs"]) == (60, 90),
        "Suite must preserve the accepted 60/90 split",
    )
    rows = suite["runs"]
    _require(isinstance(rows, list) and len(rows) == 4, "Suite must contain exactly four runs")
    _require(
        all(isinstance(row, dict) and set(row) == {
            "id", "soft_drop", "hard_drop", "entropy_beta",
        } for row in rows),
        "Every run must define only id, soft_drop, hard_drop and entropy_beta",
    )
    _require(len({row["id"] for row in rows}) == 4, "Run ids must be unique")
    observed = {
        ((float(row["soft_drop"]), float(row["hard_drop"])), float(row["entropy_beta"]))
        for row in rows
    }
    expected = {(point, beta) for point in EXPECTED_POINTS for beta in EXPECTED_BETAS}
    _require(observed == expected, "Suite must be exactly two accepted points x beta {0, 0.3}")

    verified = []
    for raw_row in rows:
        row = {
            "id": str(raw_row["id"]),
            "soft_drop": float(raw_row["soft_drop"]),
            "hard_drop": float(raw_row["hard_drop"]),
            "entropy_beta": float(raw_row["entropy_beta"]),
        }
        overrides = list(_overrides(row))
        search = schema.compose_config(suite["search_config"], overrides=overrides)
        recovery = schema.compose_config(suite["recovery_config"], overrides=overrides)
        _require(
            schema.validate_config(search) == schema.validate_config(recovery) == 150,
            f"{row['id']}: not a strict 150-epoch model",
        )
        for config in (search, recovery):
            adaptive = config.training_arguments.adaptive_lambda
            _require(
                (int(config.one_shot.search_epochs), int(config.one_shot.final_epochs))
                == (60, 90),
                f"{row['id']}: split changed",
            )
            _require(
                str(config.accuracy_guided.drop_mode) == "expected_open_count",
                f"{row['id']}: selector changed",
            )
            _require(
                str(config.one_shot.checkpoint_retention) == "online_selection",
                f"{row['id']}: checkpoint retention is not online selection",
            )
            _require(
                float(adaptive.soft_drop) == row["soft_drop"]
                and float(adaptive.hard_drop) == row["hard_drop"],
                f"{row['id']}: validation-drop point changed",
            )
            _require(
                str(config.model.entropy_regularization) == "plus_negative_entropy"
                and float(config.model.entropy_regularization_coef) == row["entropy_beta"],
                f"{row['id']}: entropy beta changed",
            )
            _require(
                float(config.accuracy_guided.eligibility.min_keep_ratio) == 0.08,
                f"{row['id']}: accepted keep floor changed",
            )
        plan = schema.resolved_branch_plan(recovery)
        _require(
            len(plan) == 1
            and plan[0]["model_state"] == "selected_surviving_state"
            and plan[0]["optimizer_state"] == "fresh"
            and plan[0]["scheduler_state"] == schema.FRESH_SCHEDULER,
            f"{row['id']}: recovery must be exactly one inherited-weight fresh-state branch",
        )
        verified.append({
            **row,
            "search_config": suite["search_config"],
            "recovery_config": suite["recovery_config"],
            "search_epochs": 60,
            "recovery_epochs": 90,
            "total_epochs": 150,
        })
    return {
        "protocol": suite["protocol"],
        "model_runs": 4,
        "runs": verified,
        "search_epochs_per_model": 60,
        "recovery_epochs_per_model": 90,
        "epochs_per_model": 150,
        "total_training_epochs": 600,
        "drop_mode": "expected_open_count",
        "checkpoint_retention": "online_selection",
        "gate_threshold_replaced": 0.5,
        "official_test_selects_nothing": True,
    }


def _launcher_command(config, dense_source, output, *, row, search_only=False,
                      reuse_search=None, skip_test=False):
    command = [
        sys.executable,
        str(ONE_SHOT),
        "--config-name", config,
        "--dense-source", str(dense_source),
        "--output", str(output),
    ]
    for override in _overrides(row):
        command.extend(["--override", override])
    if search_only:
        command.append("--search-only")
    if reuse_search is not None:
        command.extend(["--reuse-search", str(reuse_search)])
    if skip_test:
        command.append("--skip-test")
    return command


def _run(command):
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _runtime_preflight(plan, dense_source, output, min_free_gib):
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from net_complexity.training import one_shot_pruning_config as schema

    _require(min_free_gib > 0, "--min-free-gib must be positive")
    first = plan["runs"][0]
    reports = {}
    for phase in ("search", "recovery"):
        config = schema.compose_config(
            first[f"{phase}_config"],
            overrides=list(_overrides(first)),
            dense_source=dense_source,
        )
        reports[phase] = schema.validate_inputs(config)
    probe = output
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gib = usage.free / 2 ** 30
    _require(
        free_gib >= min_free_gib,
        f"Only {free_gib:.1f} GiB free on {probe}; require {min_free_gib:.1f} GiB",
    )
    return {
        "status": "ready",
        "baseline": {
            "source": str(dense_source),
            "search": reports["search"]["status"],
            "recovery": reports["recovery"]["status"],
            "initializer_model_state_hash": reports["search"].get(
                "initializer_model_state_hash"
            ),
            "reference_epochs": reports["search"].get("reference_epochs"),
        },
        "storage": {
            "filesystem_probe": str(probe),
            "free_gib": free_gib,
            "required_free_gib": min_free_gib,
        },
    }


def _verify_resolved_config(path, row):
    config = OmegaConf.load(Path(path) / "resolved_config.yaml")
    adaptive = config.training_arguments.adaptive_lambda
    _require(
        float(adaptive.soft_drop) == row["soft_drop"]
        and float(adaptive.hard_drop) == row["hard_drop"],
        f"{row['id']}: completed run has different validation drops",
    )
    _require(
        str(config.model.entropy_regularization) == "plus_negative_entropy"
        and float(config.model.entropy_regularization_coef) == row["entropy_beta"],
        f"{row['id']}: completed run has different entropy beta",
    )
    _require(
        str(config.accuracy_guided.drop_mode) == "expected_open_count",
        f"{row['id']}: completed run has a different selector",
    )


def _verify_search(path, row):
    path = Path(path)
    state = json.loads((path / "one_shot_state.json").read_text())
    selection = json.loads((path / "selection.json").read_text())
    _require(state.get("status") == "search_only_completed", "Search did not complete")
    _require(
        int(state["compute_ledger"]["actual_training_epochs_executed"]) == 60,
        "Search consumed a number of epochs other than 60",
    )
    _require(selection.get("drop_mode") == "expected_open_count", "Search used another selector")
    _verify_resolved_config(path, row)
    return {
        "status": state["status"],
        "selected_epoch": int(selection["selected_epoch"]),
        "training_epochs": 60,
    }


def _verify_recovery(path, row):
    path = Path(path)
    state = json.loads((path / "one_shot_state.json").read_text())
    _require(state.get("status") == "completed", "Recovery did not complete")
    _require(len(state.get("branches", {})) == 1, "Recovery created more than one branch")
    _require(
        int(state["compute_ledger"]["actual_training_epochs_executed"]) == 90,
        "Recovery executed a number of new epochs other than 90",
    )
    _require(
        state["selection"].get("drop_mode") == "expected_open_count",
        "Recovery reused a mask from another selector",
    )
    _verify_resolved_config(path, row)
    return {
        "status": state["status"],
        "training_epochs": 90,
        "branches": list(state["branches"]),
    }


def _prune_pair_checkpoints(search_dir, recovery_dir):
    """Remove only verified pair-local snapshots, preserving root artifacts."""
    removed_files = 0
    removed_bytes = 0
    for root in (Path(search_dir), Path(recovery_dir)):
        for path in root.rglob("checkpoints/*.pt"):
            _require(
                path.is_file() and path.parent.name == "checkpoints",
                f"Refusing unexpected cleanup target: {path}",
            )
            removed_bytes += path.stat().st_size
            path.unlink()
            removed_files += 1
    print(
        f"[suite] removed {removed_files} verified nested checkpoints "
        f"({removed_bytes / 2 ** 30:.1f} GiB); root selected/deployment artifacts retained",
        flush=True,
    )
    return {"files": removed_files, "bytes": removed_bytes}


def _write_summary(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-source", type=Path, default=DEFAULT_DENSE_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suite-config", type=Path, default=SUITE_CONFIG)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--keep-nested-checkpoints",
        action="store_true",
        help="Debug only: retain temporary per-epoch checkpoints after a verified pair",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the uploaded baseline and free disk without training",
    )
    parser.add_argument(
        "--min-free-gib", type=float, default=DEFAULT_MIN_FREE_GIB,
        help="Refuse training when less free space is available",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse completed searches/recoveries after an interrupted suite",
    )
    args = parser.parse_args(argv)

    dense_source = args.dense_source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    suite_path = args.suite_config.expanduser().resolve()
    plan = validate_suite_configs(suite_path)
    plan.update({
        "suite_config": str(suite_path),
        "dense_source": str(dense_source),
        "output": str(output),
        "official_test_after_each_frozen_recovery": not args.skip_test,
        "checkpoint_policy": (
            "retain debug snapshots instead of pruning the bounded online set"
            if args.keep_nested_checkpoints
            else "online selection retains only the current winner plus rolling last/best; "
                 "prune that bounded set after each verified pair"
        ),
    })
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return plan

    plan["preflight"] = _runtime_preflight(
        plan, dense_source, output, float(args.min_free_gib)
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return plan

    results = []
    for index, row in enumerate(plan["runs"], start=1):
        run_root = output / row["id"]
        search_output = run_root / "search"
        recovery_output = run_root / "recovery"
        print(
            f"[suite {index}/4] {row['id']} soft={row['soft_drop']:.6g} "
            f"hard={row['hard_drop']:.6g} beta={row['entropy_beta']:.1f}",
            flush=True,
        )

        if recovery_output.exists():
            if not args.resume:
                raise FileExistsError(f"Refusing existing recovery output: {recovery_output}")
            result = {
                **row,
                "search": _verify_search(search_output, row),
                "recovery": _verify_recovery(recovery_output, row),
                "reused_completed_pair": True,
            }
            if not args.keep_nested_checkpoints:
                result["checkpoint_cleanup"] = _prune_pair_checkpoints(
                    search_output, recovery_output
                )
            results.append(result)
            _write_summary(output / "study_summary.json", {
                **plan, "status": "running", "completed_runs": len(results), "results": results,
            })
            continue

        if search_output.exists():
            if not args.resume:
                raise FileExistsError(
                    f"Refusing existing search output: {search_output}; use --resume after checking it"
                )
            _verify_search(search_output, row)
        else:
            _run(_launcher_command(
                row["search_config"], dense_source, search_output,
                row=row, search_only=True,
            ))
            _verify_search(search_output, row)

        _run(_launcher_command(
            row["recovery_config"], dense_source, recovery_output,
            row=row, reuse_search=search_output, skip_test=args.skip_test,
        ))
        result = {
            **row,
            "search": _verify_search(search_output, row),
            "recovery": _verify_recovery(recovery_output, row),
            "reused_completed_pair": False,
        }
        if not args.keep_nested_checkpoints:
            result["checkpoint_cleanup"] = _prune_pair_checkpoints(
                search_output, recovery_output
            )
        results.append(result)
        _write_summary(output / "study_summary.json", {
            **plan, "status": "running", "completed_runs": len(results), "results": results,
        })

    summary = {
        **plan,
        "status": "completed",
        "completed_runs": 4,
        "results": results,
    }
    _write_summary(output / "study_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
