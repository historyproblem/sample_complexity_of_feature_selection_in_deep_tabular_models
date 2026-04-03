from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RUN_ARTIFACTS_PATTERN = re.compile(r"Run artifacts:\s*(?P<path>.+)")


@dataclass
class TrialMetrics:
    objective: float
    epoch: int
    accuracy: float
    p_mean: float
    run_dir: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_python = repo_root / ".venv" / "bin" / "python"

    parser = argparse.ArgumentParser(
        description="Run Optuna TPE search over Gumbel feature-selection hyperparameters."
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--study-name", type=str, default="gumbel_tpe_search")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--python-executable",
        type=str,
        default=str(default_python if default_python.exists() else Path(sys.executable)),
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=repo_root / "src" / "net_complexity" / "train.py",
    )
    parser.add_argument("--config-name", type=str, default="main_gumbel")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--accuracy-metric-name",
        type=str,
        default="valid_accuracy",
    )
    parser.add_argument(
        "--prob-metric-name",
        type=str,
        default="valid_average_estim_prob",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override, e.g. dataloaders.batch_size=256",
    )

    parser.add_argument("--lambda-low", type=float, default=1e-5)
    parser.add_argument("--lambda-high", type=float, default=1e-2)
    parser.add_argument("--lr-low", type=float, default=1e-4)
    parser.add_argument("--lr-high", type=float, default=3e-3)
    parser.add_argument("--temperature-low", type=float, default=0.5)
    parser.add_argument("--temperature-high", type=float, default=2.0)

    parser.add_argument("--search-init-logits", action="store_true")
    parser.add_argument("--init-on-low", type=float, default=0.5)
    parser.add_argument("--init-on-high", type=float, default=2.5)
    parser.add_argument("--init-off-low", type=float, default=0.0)
    parser.add_argument("--init-off-high", type=float, default=1.5)
    parser.add_argument("--init-noise-std", type=float, default=0.02)

    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Optional path for a JSON summary of the study result.",
    )

    return parser.parse_args()


def read_best_metrics(
    history_path: Path,
    beta: float,
    accuracy_metric_name: str,
    prob_metric_name: str,
    run_dir: Path,
) -> TrialMetrics:
    best_metrics: TrialMetrics | None = None

    with history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            accuracy = float(row[accuracy_metric_name])
            p_mean = float(row[prob_metric_name])
            objective = (1.0 - accuracy) + beta * p_mean
            metrics = TrialMetrics(
                objective=objective,
                epoch=int(row["epoch"]),
                accuracy=accuracy,
                p_mean=p_mean,
                run_dir=str(run_dir),
            )
            if best_metrics is None or metrics.objective < best_metrics.objective:
                best_metrics = metrics

    if best_metrics is None:
        raise ValueError(f"No epoch metrics found in {history_path}")

    return best_metrics


def extract_run_dir(stdout: str, stderr: str) -> Path:
    for stream in (stdout, stderr):
        matches = RUN_ARTIFACTS_PATTERN.findall(stream)
        if matches:
            return Path(matches[-1].strip())
    raise ValueError("Could not determine run directory from train output.")


def run_trial(
    trial,
    args: argparse.Namespace,
    repo_root: Path,
) -> float:
    lambda_coef = trial.suggest_float("lambda_coef", args.lambda_low, args.lambda_high, log=True)
    lr = trial.suggest_float("lr", args.lr_low, args.lr_high, log=True)
    temperature = trial.suggest_float(
        "temperature",
        args.temperature_low,
        args.temperature_high,
    )

    overrides = [
        "hydra.job.chdir=false",
        f"model.lambda_coef={lambda_coef}",
        f"optimizer.lr={lr}",
        f"++model.backbone.resnet_block.temperature={temperature}",
        f"mlflow.run_name=optuna_trial_{trial.number}",
        f"++run_history.run_name=optuna_trial_{trial.number}",
        "mlflow.log_model=false",
        "mlflow.log_artifacts=false",
    ]

    if args.search_init_logits:
        init_on = trial.suggest_float("init_on", args.init_on_low, args.init_on_high)
        init_off = trial.suggest_float("init_off", args.init_off_low, args.init_off_high)
        overrides.extend(
            [
                f"++model.backbone.resnet_block.init_on={init_on}",
                f"++model.backbone.resnet_block.init_off={init_off}",
                f"++model.backbone.resnet_block.init_noise_std={args.init_noise_std}",
            ]
        )

    if args.device is not None:
        overrides.append(f"device={args.device}")
    if args.epochs is not None:
        overrides.append(f"training_arguments.num_epochs={args.epochs}")
    overrides.extend(args.override)

    command = [args.python_executable, str(args.train_script)]
    if args.config_name != "main_gumbel":
        command.extend(["--config-name", args.config_name])
    command.extend(overrides)

    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(
            part for part in (completed.stdout[-4000:], completed.stderr[-4000:]) if part
        )
        raise RuntimeError(
            f"Trial {trial.number} failed with exit code {completed.returncode}.\n{tail}"
        )

    run_dir = extract_run_dir(completed.stdout, completed.stderr)
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        raise ValueError(f"Missing history file: {history_path}")

    best_metrics = read_best_metrics(
        history_path=history_path,
        beta=args.beta,
        accuracy_metric_name=args.accuracy_metric_name,
        prob_metric_name=args.prob_metric_name,
        run_dir=run_dir,
    )

    trial.set_user_attr("run_dir", best_metrics.run_dir)
    trial.set_user_attr("best_epoch", best_metrics.epoch)
    trial.set_user_attr("best_accuracy", best_metrics.accuracy)
    trial.set_user_attr("best_p_mean", best_metrics.p_mean)
    trial.set_user_attr("best_objective", best_metrics.objective)

    return best_metrics.objective


def make_summary(study, args: argparse.Namespace) -> dict:
    best_trial = study.best_trial
    return {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "beta": args.beta,
        "n_trials": len(study.trials),
        "best_value": best_trial.value,
        "best_params": dict(best_trial.params),
        "best_user_attrs": dict(best_trial.user_attrs),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed in the active environment. "
            "Install it first, then rerun tune_gumbel.py."
        ) from exc

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(
        lambda trial: run_trial(trial, args, repo_root),
        n_trials=args.n_trials,
        catch=(RuntimeError, ValueError, subprocess.SubprocessError),
    )

    completed_trials = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        print("No trial completed successfully.", file=sys.stderr)
        return 1

    summary = make_summary(study, args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.summary_path is not None:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
