"""Run six target-5M searches, select one on validation, and recover it twice."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import time


ROOT = Path(__file__).resolve().parents[1]
ONE_SHOT = ROOT / "scripts/launch_one_shot_pruning.py"
RECOVERY_CONFIG = "experiment/pruning_v3/target5m_mapped_recovery_repeats2"
SEARCH_CONFIGS = (
    "search60_soft0p5_crit1p0_fixed",
    "search60_soft0p5_crit1p0_auto",
    "search60_soft0p75_crit1p0_fixed",
    "search60_soft0p75_crit1p0_auto",
    "search60_soft1p0_crit1p5_fixed",
    "search60_soft1p0_crit1p5_auto",
)
SEARCH_DROPS = {
    "search60_soft0p5_crit1p0_fixed": (0.005, 0.010),
    "search60_soft0p5_crit1p0_auto": (0.005, 0.010),
    "search60_soft0p75_crit1p0_fixed": (0.0075, 0.010),
    "search60_soft0p75_crit1p0_auto": (0.0075, 0.010),
    "search60_soft1p0_crit1p5_fixed": (0.010, 0.015),
    "search60_soft1p0_crit1p5_auto": (0.010, 0.015),
}


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _run(command, *, capture=False):
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if result.returncode:
        detail = f"\n{result.stdout.rstrip()}" if capture and result.stdout else ""
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(map(str, command))}{detail}"
        )
    return result


def _launcher_command(config, dense_source, output, *, min_keep_ratio=None, search_only=False,
                      reuse_search=None, dry_run=False, overrides=()):
    command = [
        sys.executable,
        str(ONE_SHOT),
        "--config-name", config,
        "--dense-source", str(dense_source),
        "--output", str(output),
    ]
    if min_keep_ratio is not None:
        overrides = (
            *overrides,
            f"accuracy_guided.eligibility.min_keep_ratio={min_keep_ratio}",
        )
    for override in overrides:
        command.extend(["--override", override])
    if search_only:
        command.append("--search-only")
    if reuse_search is not None:
        command.extend(["--reuse-search", str(reuse_search)])
    if dry_run:
        command.append("--dry-run")
    return command


def _search_result(run_dir):
    state_path = run_dir / "one_shot_state.json"
    diagnostics_path = run_dir / "export_only/diagnostics.json"
    selection_path = run_dir / "selection.json"
    checkpoint_path = run_dir / "selected_checkpoint.pt"
    for path in (state_path, diagnostics_path, selection_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"Completed search is missing {path}")
    state = json.loads(state_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    selection = json.loads(selection_path.read_text())
    if state.get("status") != "search_only_completed":
        raise RuntimeError(f"Search did not complete: {run_dir} status={state.get('status')}")
    if int(state["shared_search_ledger"]["search_epochs_consumed"]) != 60:
        raise RuntimeError(f"Search did not consume 60 epochs: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "selected_epoch": int(selection["selected_epoch"]),
        "gated_validation_accuracy": float(diagnostics["gated_validation"]["accuracy"]),
        "gated_validation_ce_loss": float(diagnostics["gated_validation"]["ce_loss"]),
        "physical_validation_accuracy": float(diagnostics["physical_validation"]["accuracy"]),
        "physical_validation_ce_loss": float(diagnostics["physical_validation"]["ce_loss"]),
        "physical_parameters": int(diagnostics["physical_cost"]["physical_total_parameters"]),
        "physical_macs": int(diagnostics["physical_cost"]["conv_linear_macs_per_image"]),
        "blocked_closed_survivors": len(diagnostics["blocked_closed_survivors_opened_by_export"]),
        "gated_equivalent_on_checked_batch": bool(diagnostics["gated_equivalent_on_checked_batch"]),
    }


def select_winner(results, target_parameters):
    within_target = [row for row in results if row["physical_parameters"] <= target_parameters]
    if within_target:
        winner = max(
            within_target,
            key=lambda row: (row["physical_validation_accuracy"], row["physical_parameters"]),
        )
        policy = "highest_physical_validation_accuracy_at_or_below_target"
    else:
        winner = min(
            results,
            key=lambda row: (row["physical_parameters"], -row["physical_validation_accuracy"]),
        )
        policy = "smallest_physical_model_when_no_candidate_met_target"
    return winner, policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runs/target5m_nightly_keep0p14"),
    )
    parser.add_argument("--min-keep-ratio", type=float, default=0.14)
    parser.add_argument("--target-parameters", type=int, default=5_000_000)
    args = parser.parse_args(argv)
    if not 0 < args.min_keep_ratio <= 1:
        parser.error("--min-keep-ratio must be in (0, 1]")
    if args.target_parameters <= 0:
        parser.error("--target-parameters must be positive")

    dense_source = args.dense_source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing nightly output: {output}. Use a fresh --output directory.")

    # Fail before spending GPU hours if CUDA, dependencies, configs or immutable
    # dense-reference inputs are unavailable.
    _run([
        sys.executable,
        "-c",
        "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))",
    ])
    for config_name in SEARCH_CONFIGS:
        _run(_launcher_command(
            f"experiment/pruning_v3/{config_name}",
            dense_source,
            output / "searches" / config_name,
            min_keep_ratio=args.min_keep_ratio,
            search_only=True,
            dry_run=True,
        ), capture=True)
    _run(_launcher_command(
        RECOVERY_CONFIG,
        dense_source,
        output / "winner_recovery_repeats2",
        reuse_search=output / "searches" / SEARCH_CONFIGS[0],
        dry_run=True,
    ), capture=True)

    output.mkdir(parents=True)
    state_path = output / "nightly_state.json"
    state = {
        "protocol": "target5m_nightly_six_searches_then_mapped_recovery_repeats2",
        "status": "running",
        "started_unix": time(),
        "dense_source": str(dense_source),
        "min_keep_ratio": args.min_keep_ratio,
        "target_parameters": args.target_parameters,
        "search_configs": list(SEARCH_CONFIGS),
        "searches": {},
        "selection": None,
        "recovery": None,
    }
    _write_json(state_path, state)
    try:
        results = []
        for index, config_name in enumerate(SEARCH_CONFIGS, 1):
            run_dir = output / "searches" / config_name
            print(f"[nightly] search {index}/{len(SEARCH_CONFIGS)}: {config_name}", flush=True)
            _run(_launcher_command(
                f"experiment/pruning_v3/{config_name}",
                dense_source,
                run_dir,
                min_keep_ratio=args.min_keep_ratio,
                search_only=True,
            ))
            soft_drop, critical_drop = SEARCH_DROPS[config_name]
            result = {
                "config": config_name,
                "soft_drop": soft_drop,
                "critical_drop": critical_drop,
                **_search_result(run_dir),
            }
            results.append(result)
            state["searches"][config_name] = result
            _write_json(state_path, state)

        winner, policy = select_winner(results, args.target_parameters)
        state["selection"] = {"policy": policy, "winner": winner}
        _write_json(state_path, state)
        print(
            f"[nightly] winner={winner['config']} params={winner['physical_parameters']:,} "
            f"physical_val={winner['physical_validation_accuracy']:.4%}",
            flush=True,
        )

        recovery_dir = output / "winner_recovery_repeats2"
        _run(_launcher_command(
            RECOVERY_CONFIG,
            dense_source,
            recovery_dir,
            reuse_search=Path(winner["run_dir"]),
            overrides=(
                f"training_arguments.adaptive_lambda.soft_drop={winner['soft_drop']}",
                f"training_arguments.adaptive_lambda.hard_drop={winner['critical_drop']}",
            ),
        ))
        recovery_state = json.loads((recovery_dir / "one_shot_state.json").read_text())
        if recovery_state.get("status") != "completed":
            raise RuntimeError(f"Recovery did not complete: status={recovery_state.get('status')}")
        state["recovery"] = {
            "run_dir": str(recovery_dir),
            "status": recovery_state["status"],
            "branches": {
                name: {
                    "training_seed": branch["training_seed"],
                    "validation_accuracy": branch["validation"]["accuracy"],
                    "physical_parameters": branch["final_cost"]["physical_total_parameters"],
                    "quality_feasible": branch["quality_feasible"],
                }
                for name, branch in recovery_state["branches"].items()
            },
            "official_test_summary": str(recovery_dir / "test_evaluation/test_summary.json"),
        }
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
