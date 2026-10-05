"""Run four new expected-open L1/L2 ResNet50/CIFAR-10 models.

The complete comparison is three accepted 60/90 adaptive-lambda drop pairs
crossed with L1/L2 gate-probability penalties. The preceding expected-open
suite already completed the L1 models for two of those pairs, so this launcher
runs only the four missing models: L1 for (0.00025, 0.0005) and L2 for all
three pairs. Every new model receives exactly 60 search plus 90 recovery epochs.
The existing matched dense reference is required and is never retrained here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import launch_resnet50_expected_open_four_runs as common


ROOT = common.ROOT
ONE_SHOT = common.ONE_SHOT
SUITE_CONFIG = (
    ROOT
    / "configs/experiment/pruning_v3/resnet50_cifar10_expected_open_l1_l2.yaml"
)
DEFAULT_DENSE_SOURCE = common.DEFAULT_DENSE_SOURCE
DEFAULT_OUTPUT = Path(
    "outputs/runs/resnet50_cifar10_expected_open_l1_l2_four_new_v1"
)
EXPECTED_NEW_RUNS = {
    ((0.00025, 0.0005), "l1_probability"),
    ((0.00025, 0.0005), "l2_probability"),
    ((0.0005, 0.001), "l2_probability"),
    ((0.01, 0.02), "l2_probability"),
}
EXPECTED_SKIPPED_RUNS = {
    ((0.0005, 0.001), "l1_probability", "target12m_beta0"),
    ((0.01, 0.02), "l1_probability", "target5m_beta0"),
}
MIN_FREE_GIB = 10.0


def _plain_suite(path=SUITE_CONFIG):
    suite = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    common._require(isinstance(suite, dict), "Suite config must be a mapping")
    common._require(
        set(suite) == {
            "protocol", "baseline", "search_config", "recovery_config",
            "drop_mode", "search_epochs", "recovery_epochs",
            "skipped_completed_runs", "runs",
        },
        "Suite config fields changed",
    )
    return suite


def _overrides(row):
    return (
        f"training_arguments.adaptive_lambda.soft_drop={row['soft_drop']:.12g}",
        f"training_arguments.adaptive_lambda.hard_drop={row['hard_drop']:.12g}",
        "model.entropy_regularization=disabled",
        "model.entropy_regularization_coef=0",
        "model.backbone.resnet_block.gate_regularization="
        f"{row['gate_regularization']}",
    )


def validate_suite_configs(path=SUITE_CONFIG):
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from net_complexity.training import one_shot_pruning_config as schema

    suite = _plain_suite(path)
    common._require(
        suite["protocol"] == "resnet50_cifar10_expected_open_l1_l2_v1",
        "Unexpected suite protocol",
    )
    common._require(suite["drop_mode"] == "expected_open_count", "Unexpected selector")
    common._require(
        (suite["search_epochs"], suite["recovery_epochs"]) == (60, 90),
        "Suite must preserve the accepted 60/90 split",
    )
    baseline = suite["baseline"]
    common._require(
        baseline == {
            "config": suite["search_config"],
            "epochs": 150,
            "seed": 42,
            "checkpoint_retention": "metadata_only",
            "prepare_if_missing": False,
        },
        "Suite must reuse the matched seed-42 dense150 reference",
    )

    rows = suite["runs"]
    common._require(isinstance(rows, list) and len(rows) == 4,
                    "Suite must contain exactly four new runs")
    expected_fields = {"id", "soft_drop", "hard_drop", "gate_regularization"}
    common._require(
        all(isinstance(row, dict) and set(row) == expected_fields for row in rows),
        "Every new run must define id, drops and gate_regularization",
    )
    observed = {
        ((float(row["soft_drop"]), float(row["hard_drop"])),
         str(row["gate_regularization"]))
        for row in rows
    }
    common._require(observed == EXPECTED_NEW_RUNS,
                    "Suite does not contain the exact four missing L1/L2 runs")
    common._require(len({row["id"] for row in rows}) == 4, "Run ids must be unique")

    skipped = suite["skipped_completed_runs"]
    common._require(isinstance(skipped, list) and len(skipped) == 2,
                    "Exactly two completed L1 runs must be skipped")
    skipped_observed = {
        ((float(row["soft_drop"]), float(row["hard_drop"])),
         str(row["gate_regularization"]), str(row["prior_suite_id"]))
        for row in skipped
    }
    common._require(skipped_observed == EXPECTED_SKIPPED_RUNS,
                    "Skipped controls differ from the completed beta=0 suite")

    baseline_config = schema.compose_config(baseline["config"])
    common._require(
        schema.validate_config(baseline_config) == 150
        and int(baseline_config.seed) == 42
        and str(baseline_config.one_shot.reference_checkpoint_retention)
        == "metadata_only",
        "Baseline config differs from the matched dense150 contract",
    )

    verified = []
    for raw_row in rows:
        row = {
            "id": str(raw_row["id"]),
            "soft_drop": float(raw_row["soft_drop"]),
            "hard_drop": float(raw_row["hard_drop"]),
            "gate_regularization": str(raw_row["gate_regularization"]),
        }
        overrides = list(_overrides(row))
        search = schema.compose_config(suite["search_config"], overrides=overrides)
        recovery = schema.compose_config(suite["recovery_config"], overrides=overrides)
        common._require(
            schema.validate_config(search) == schema.validate_config(recovery) == 150,
            f"{row['id']}: not a strict 150-epoch model",
        )
        for config in (search, recovery):
            adaptive = config.training_arguments.adaptive_lambda
            block = config.model.backbone.resnet_block
            common._require(
                (int(config.one_shot.search_epochs), int(config.one_shot.final_epochs))
                == (60, 90),
                f"{row['id']}: split changed",
            )
            common._require(
                str(config.accuracy_guided.drop_mode) == "expected_open_count",
                f"{row['id']}: expected-open selector changed",
            )
            common._require(
                str(config.one_shot.checkpoint_retention) == "online_selection",
                f"{row['id']}: checkpoint retention changed",
            )
            common._require(
                float(adaptive.soft_drop) == row["soft_drop"]
                and float(adaptive.hard_drop) == row["hard_drop"],
                f"{row['id']}: adaptive-lambda drops changed",
            )
            common._require(
                str(block.gate_regularization) == row["gate_regularization"],
                f"{row['id']}: gate regularization changed",
            )
            common._require(
                str(config.model.entropy_regularization) == "disabled"
                and float(config.model.entropy_regularization_coef) == 0.0,
                f"{row['id']}: entropy must be disabled for the L1/L2 comparison",
            )
            common._require(
                float(config.accuracy_guided.eligibility.min_keep_ratio) == 0.08,
                f"{row['id']}: keep floor changed",
            )
        plan = schema.resolved_branch_plan(recovery)
        common._require(
            len(plan) == 1
            and plan[0]["model_state"] == "selected_surviving_state"
            and plan[0]["optimizer_state"] == "fresh"
            and plan[0]["scheduler_state"] == schema.FRESH_SCHEDULER,
            f"{row['id']}: recovery must be one inherited-weight fresh-state branch",
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
        "baseline": dict(baseline),
        "model_runs": 4,
        "runs": verified,
        "skipped_completed_runs": skipped,
        "search_epochs_per_model": 60,
        "recovery_epochs_per_model": 90,
        "epochs_per_model": 150,
        "total_new_training_epochs": 600,
        "drop_mode": "expected_open_count",
        "gate_regularization": {
            "l1_probability": "mean(p_open)",
            "l2_probability": "mean(p_open^2)",
        },
        "entropy_regularization": "disabled",
        "checkpoint_retention": "online_selection",
        "official_test_selects_nothing": True,
    }


def _launcher_command(config, dense_source, output, *, row, search_only=False,
                      reuse_search=None):
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
    return command


def _runtime_preflight(plan, dense_source, output):
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from net_complexity.training import one_shot_pruning_config as schema

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
    free_gib = shutil.disk_usage(probe).free / 2 ** 30
    common._require(
        free_gib >= MIN_FREE_GIB,
        f"Only {free_gib:.1f} GiB free on {probe}; require {MIN_FREE_GIB:.1f} GiB",
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
            "required_free_gib": MIN_FREE_GIB,
        },
    }


def _verify_resolved_config(path, row):
    config = OmegaConf.load(Path(path) / "resolved_config.yaml")
    adaptive = config.training_arguments.adaptive_lambda
    common._require(
        float(adaptive.soft_drop) == row["soft_drop"]
        and float(adaptive.hard_drop) == row["hard_drop"],
        f"{row['id']}: completed run has different adaptive-lambda drops",
    )
    common._require(
        str(config.model.backbone.resnet_block.gate_regularization)
        == row["gate_regularization"],
        f"{row['id']}: completed run has different gate regularization",
    )
    common._require(
        str(config.model.entropy_regularization) == "disabled"
        and float(config.model.entropy_regularization_coef) == 0.0,
        f"{row['id']}: completed run unexpectedly used entropy",
    )
    common._require(
        str(config.accuracy_guided.drop_mode) == "expected_open_count",
        f"{row['id']}: completed run used another selector",
    )


def _verify_search(path, row):
    path = Path(path)
    state = json.loads((path / "one_shot_state.json").read_text())
    selection = json.loads((path / "selection.json").read_text())
    common._require(state.get("status") == "search_only_completed",
                    "Search did not complete")
    common._require(
        int(state["compute_ledger"]["actual_training_epochs_executed"]) == 60,
        "Search consumed a number of epochs other than 60",
    )
    common._require(selection.get("drop_mode") == "expected_open_count",
                    "Search used another selector")
    _verify_resolved_config(path, row)
    return {
        "status": state["status"],
        "selected_epoch": int(selection["selected_epoch"]),
        "training_epochs": 60,
    }


def _verify_recovery(path, row):
    path = Path(path)
    state = json.loads((path / "one_shot_state.json").read_text())
    common._require(state.get("status") == "completed", "Recovery did not complete")
    common._require(len(state.get("branches", {})) == 1,
                    "Recovery created more than one branch")
    common._require(
        int(state["compute_ledger"]["actual_training_epochs_executed"]) == 90,
        "Recovery executed a number of new epochs other than 90",
    )
    common._require(state["selection"].get("drop_mode") == "expected_open_count",
                    "Recovery reused a mask from another selector")
    _verify_resolved_config(path, row)
    return {
        "status": state["status"],
        "training_epochs": 90,
        "branches": list(state["branches"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Verify and continue this fixed suite after an interruption",
    )
    args = parser.parse_args(argv)

    dense_source = DEFAULT_DENSE_SOURCE.expanduser().resolve()
    output = DEFAULT_OUTPUT.expanduser().resolve()
    plan = validate_suite_configs()
    plan.update({
        "suite_config": str(SUITE_CONFIG.resolve()),
        "dense_source": str(dense_source),
        "output": str(output),
        "baseline_action": "reuse" if dense_source.exists() else "refuse_missing",
        "official_test_after_each_frozen_recovery": True,
        "checkpoint_policy": (
            "online selection retains a bounded candidate set; nested checkpoints "
            "are removed after each verified search/recovery pair"
        ),
    })
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return plan

    common._require(
        dense_source.exists(),
        f"Matched dense reference is missing: {dense_source}. "
        "Refusing to add an unrequested fifth training job.",
    )

    plan["preflight"] = _runtime_preflight(plan, dense_source, output)
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)

    results = []
    for index, row in enumerate(plan["runs"], start=1):
        run_root = output / row["id"]
        search_output = run_root / "search"
        recovery_output = run_root / "recovery"
        print(
            f"[suite {index}/4] {row['id']} soft={row['soft_drop']:.6g} "
            f"hard={row['hard_drop']:.6g} reg={row['gate_regularization']}",
            flush=True,
        )

        if recovery_output.exists():
            if not args.resume:
                raise FileExistsError(
                    f"Refusing existing recovery output: {recovery_output}"
                )
            result = {
                **row,
                "search": _verify_search(search_output, row),
                "recovery": _verify_recovery(recovery_output, row),
                "reused_completed_pair": True,
            }
        else:
            if search_output.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"Refusing existing search output: {search_output}; "
                        "use --resume after checking it"
                    )
                _verify_search(search_output, row)
            else:
                common._run(_launcher_command(
                    row["search_config"], dense_source, search_output,
                    row=row, search_only=True,
                ))
                _verify_search(search_output, row)

            common._run(_launcher_command(
                row["recovery_config"], dense_source, recovery_output,
                row=row, reuse_search=search_output,
            ))
            result = {
                **row,
                "search": _verify_search(search_output, row),
                "recovery": _verify_recovery(recovery_output, row),
                "reused_completed_pair": False,
            }

        result["checkpoint_cleanup"] = common._prune_pair_checkpoints(
            search_output, recovery_output
        )
        results.append(result)
        common._write_summary(output / "study_summary.json", {
            **plan,
            "status": "running",
            "completed_runs": len(results),
            "results": results,
        })

    summary = {
        **plan,
        "status": "completed",
        "completed_runs": 4,
        "results": results,
    }
    common._write_summary(output / "study_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
