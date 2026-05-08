from pathlib import Path

import pandas as pd
import yaml


STUDIES_ROOT = Path("PUT_YOUR_STUDIES_ROOT_HERE")
CONFIG_NAME = "config_resolved.yaml"


def flatten_dict(d: dict, prefix: str = "") -> dict:
    result = {}

    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_dict(value, full_key))
        else:
            result[full_key] = value

    return result


def get_last_exp(studies_root: Path = STUDIES_ROOT) -> Path:
    exp_dirs = sorted(
        p for p in studies_root.iterdir()
        if p.is_dir() and (p / "runs").is_dir()
    )

    if not exp_dirs:
        raise ValueError(f"No experiments with runs/ found in: {studies_root}")

    return exp_dirs[-1]


def get_lambda_coef(cfg: dict) -> float | None:
    warmup = cfg.get("training_arguments", {}).get("lambda_warmup", {})

    if warmup.get("enabled", False):
        value = warmup.get("target_lambda_coef")
    else:
        value = cfg.get("model", {}).get("lambda_coef")

    return None if value is None else float(value)


def perform_last_exp(
    studies_root: Path = STUDIES_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_dir = get_last_exp(studies_root)
    runs_dir = exp_dir / "runs"

    history_parts = []
    config_rows = []

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        history_path = run_dir / "history.csv"
        config_path = run_dir / CONFIG_NAME

        if not history_path.exists():
            print(f"skip {run_dir.name}: no history.csv")
            continue

        if not config_path.exists():
            print(f"skip {run_dir.name}: no {CONFIG_NAME}")
            continue

        history = pd.read_csv(history_path)

        if history.empty:
            print(f"skip {run_dir.name}: empty history.csv")
            continue

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        flat_cfg = flatten_dict(cfg)
        lambda_coef = get_lambda_coef(cfg)

        base_info = {
            "experiment_name": exp_dir.name,
            "experiment_dir": str(exp_dir),
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "history_path": str(history_path),
            "config_path": str(config_path),
            "lambda_coef": lambda_coef,
            "lambda_str": "unknown" if lambda_coef is None else f"{lambda_coef:g}",
        }

        history = history.copy()

        for key, value in reversed(base_info.items()):
            history.insert(0, key, value)

        history_parts.append(history)

        config_rows.append({
            **base_info,
            **flat_cfg,
        })

    if not history_parts:
        raise ValueError(f"No valid runs found in: {runs_dir}")

    df_history = pd.concat(history_parts, ignore_index=True)
    df_config = pd.DataFrame(config_rows)

    print("experiment:", exp_dir.name)
    print("runs loaded:", df_history["run_name"].nunique())
    print("history rows:", len(df_history))

    return df_history, df_config


def make_summary(df_history: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for run_name, g in df_history.groupby("run_name"):
        g = g.sort_values("epoch")

        best_row = g.loc[g["valid_accuracy"].idxmax()]
        final_row = g.iloc[-1]

        rows.append({
            "run_name": run_name,
            "lambda_coef": final_row["lambda_coef"],
            "lambda_str": final_row["lambda_str"],
            "best_epoch": int(best_row["epoch"]),
            "best_valid_accuracy": float(best_row["valid_accuracy"]),
            "final_epoch": int(final_row["epoch"]),
            "final_valid_accuracy": float(final_row["valid_accuracy"]),
            "num_epochs": int(g["epoch"].nunique()),
            "run_dir": final_row["run_dir"],
        })

    return (
        pd.DataFrame(rows)
        .sort_values("best_valid_accuracy", ascending=False)
        .reset_index(drop=True)
    )
