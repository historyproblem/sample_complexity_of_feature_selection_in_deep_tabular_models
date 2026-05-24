from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


STUDIES_ROOT = Path("PUT_YOUR_STUDIES_ROOT_HERE")

CONFIG_NAMES = [
    "config_resolved.yaml",
    "configs_resolved.yaml",
    "resolved_config.yaml",
]


META_COLS = [
    "experiment_name",
    "experiment_dir",
    "run_name",
    "run_dir",
    "history_path",
    "config_path",
    "config_lambda_coef",
    "config_lambda_str",
    "run_label",
]


# =========================
# CONFIG UTILS
# =========================

def flatten_dict(d: dict, prefix: str = "") -> dict:
    result = {}

    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_dict(value, full_key))
        else:
            result[full_key] = value

    return result


def find_config_path(run_dir: Path) -> Path | None:
    for name in CONFIG_NAMES:
        path = run_dir / name
        if path.exists():
            return path

    return None


def read_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_config_lambda_coef(cfg: dict) -> float | None:
    warmup = cfg.get("training_arguments", {}).get("lambda_warmup", {})

    if warmup.get("enabled", False):
        value = warmup.get("target_lambda_coef")
    else:
        value = cfg.get("model", {}).get("lambda_coef")

    return None if value is None else float(value)


# backward-compatible alias
def get_lambda_coef(cfg: dict) -> float | None:
    return get_config_lambda_coef(cfg)


# =========================
# HISTORY NORMALIZATION
# =========================

def _format_float(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "unknown"
    return f"{float(x):g}"


def make_run_label(
    history: pd.DataFrame,
    run_name: str,
    config_lambda_coef: float | None,
) -> str:
    """
    Статический label для легенды.

    Важно:
    - lambda_coef в history может меняться по эпохам;
    - поэтому нельзя использовать per-row lambda_coef как lambda_str для groupby.
    """
    if config_lambda_coef is not None:
        return f"λ={config_lambda_coef:g}"

    if "lambda_coef" in history.columns:
        lambdas = history["lambda_coef"].dropna()

        if len(lambdas) > 0:
            first = float(lambdas.iloc[0])
            last = float(lambdas.iloc[-1])

            if np.isclose(first, last):
                return f"λ={first:g}"

            return f"adaptive λ: {first:g}→{last:g}"

    return run_name


def normalize_history_columns(
    history: pd.DataFrame,
    config_lambda_coef: float | None = None,
) -> pd.DataFrame:
    """
    Приводит разные версии history.csv к удобному виду.

    Не перезаписывает реальные столбцы результата:
    - lambda_coef
    - valid_accuracy
    - valid_average_zero_prob
    - valid_real_active_channels
    - valid_estim_active_channels
    и т.д.
    """
    history = history.copy()

    # epoch
    if "epoch" in history.columns:
        history["epoch"] = pd.to_numeric(history["epoch"], errors="coerce")

    # lambda_coef: берем из таблицы, если он там есть.
    # Если нет — fallback из model_lambda_coef, log_lambda или config.
    if "lambda_coef" not in history.columns:
        if "model_lambda_coef" in history.columns:
            history["lambda_coef"] = history["model_lambda_coef"]
        elif "log_lambda" in history.columns:
            history["lambda_coef"] = np.exp(history["log_lambda"])
        elif config_lambda_coef is not None:
            history["lambda_coef"] = config_lambda_coef

    # lambda_str должен быть статическим label-ом, не per-epoch lambda.
    # Иначе adaptive lambda порежет одну линию на десятки маленьких линий.
    if "lambda_str" in history.columns:
        history = history.drop(columns=["lambda_str"])

    # Стандартные aliases для графиков.
    if "valid_average_zero_prob" in history.columns:
        history["valid_active_ratio"] = 1.0 - history["valid_average_zero_prob"]
        history["valid_zero_ratio"] = history["valid_average_zero_prob"]

    if "valid_real_active_channels" in history.columns:
        history["valid_active_channels"] = history["valid_real_active_channels"]

    if "valid_real_zero_channels" in history.columns:
        history["valid_zero_channels"] = history["valid_real_zero_channels"]

    if "valid_estim_active_channels" in history.columns:
        history["valid_estim_active_channels_alias"] = history["valid_estim_active_channels"]

    if "valid_estim_zero_channels" in history.columns:
        history["valid_estim_zero_channels_alias"] = history["valid_estim_zero_channels"]

    # Если real channels нет, используем estim.
    if "valid_active_channels" not in history.columns:
        if "valid_estim_active_channels" in history.columns:
            history["valid_active_channels"] = history["valid_estim_active_channels"]

    if "valid_zero_channels" not in history.columns:
        if "valid_estim_zero_channels" in history.columns:
            history["valid_zero_channels"] = history["valid_estim_zero_channels"]

    # train aliases
    if "train_average_zero_prob" in history.columns:
        history["train_active_ratio"] = 1.0 - history["train_average_zero_prob"]
        history["train_zero_ratio"] = history["train_average_zero_prob"]

    if "train_real_active_channels" in history.columns:
        history["train_active_channels"] = history["train_real_active_channels"]

    if "train_real_zero_channels" in history.columns:
        history["train_zero_channels"] = history["train_real_zero_channels"]

    return history


def add_meta_columns(history: pd.DataFrame, base_info: dict) -> pd.DataFrame:
    """
    Добавляет meta columns, но не затирает результатные колонки history.csv.
    """
    history = history.copy()

    for key, value in base_info.items():
        if key in history.columns:
            history[f"meta_{key}"] = value
        else:
            history[key] = value

    other_cols = [c for c in history.columns if c not in META_COLS]
    existing_meta_cols = [c for c in META_COLS if c in history.columns]

    return history[existing_meta_cols + other_cols]


# =========================
# LOADING
# =========================

def get_last_exp(studies_root: Path = STUDIES_ROOT) -> Path:
    studies_root = Path(studies_root)

    exp_dirs = sorted(
        p for p in studies_root.iterdir()
        if p.is_dir() and (p / "runs").is_dir()
    )

    if not exp_dirs:
        raise ValueError(f"No experiments with runs/ found in: {studies_root}")

    return exp_dirs[-1]


def load_single_run(
    run_dir: Path,
    experiment_name: str | None = None,
    experiment_dir: Path | None = None,
    run_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    run_dir = Path(run_dir)

    history_path = run_dir / "history.csv"

    if not history_path.exists():
        raise ValueError(f"No history.csv found in: {run_dir}")

    history = pd.read_csv(history_path)

    if history.empty:
        raise ValueError(f"Empty history.csv: {history_path}")

    config_path = find_config_path(run_dir)
    cfg = read_config(config_path)

    flat_cfg = flatten_dict(cfg)
    config_lambda_coef = get_config_lambda_coef(cfg)

    history = normalize_history_columns(
        history,
        config_lambda_coef=config_lambda_coef,
    )

    if experiment_dir is None:
        experiment_dir = run_dir.parent

    if experiment_name is None:
        experiment_name = Path(experiment_dir).name

    if run_name is None:
        run_name = run_dir.name

    run_label = make_run_label(
        history=history,
        run_name=run_name,
        config_lambda_coef=config_lambda_coef,
    )

    # Статическая строка для старых функций, которые ждут lambda_str.
    history["lambda_str"] = run_label

    base_info = {
        "experiment_name": experiment_name,
        "experiment_dir": str(experiment_dir),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "history_path": str(history_path),
        "config_path": None if config_path is None else str(config_path),
        "config_lambda_coef": config_lambda_coef,
        "config_lambda_str": "unknown" if config_lambda_coef is None else f"{config_lambda_coef:g}",
        "run_label": run_label,
    }

    history = add_meta_columns(history, base_info)

    config_row = {
        **base_info,
        **flat_cfg,
    }

    return history, config_row


def load_direct_history_dir(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Случай:
        path/
            history.csv
            config_resolved.yaml  # optional
    """
    path = Path(path)

    history, config_row = load_single_run(
        run_dir=path,
        experiment_name=path.name,
        experiment_dir=path,
        run_name=path.name,
    )

    df_history = history.reset_index(drop=True)
    df_config = pd.DataFrame([config_row])

    return df_history, df_config


def load_experiment_dir(exp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Случай:
        exp_dir/
            runs/
                run_1/
                    history.csv
                    config_resolved.yaml
                run_2/
                    history.csv
                    config_resolved.yaml
    """
    exp_dir = Path(exp_dir)
    runs_dir = exp_dir / "runs"

    if not runs_dir.is_dir():
        raise ValueError(f"No runs/ found in: {exp_dir}")

    history_parts = []
    config_rows = []

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        try:
            history, config_row = load_single_run(
                run_dir=run_dir,
                experiment_name=exp_dir.name,
                experiment_dir=exp_dir,
                run_name=run_dir.name,
            )
        except ValueError as e:
            print(f"skip {run_dir.name}: {e}")
            continue

        history_parts.append(history)
        config_rows.append(config_row)

    if not history_parts:
        raise ValueError(f"No valid runs found in: {runs_dir}")

    df_history = pd.concat(history_parts, ignore_index=True)
    df_config = pd.DataFrame(config_rows)

    return df_history, df_config


def perform_exp(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Универсальный загрузчик.

    Поддерживает:
    1. path/history.csv
    2. path/runs/run_x/history.csv
    3. path как studies_root, где внутри есть exp_x/runs/
    4. прямой путь до history.csv
    """
    path = Path(path)

    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    # Case 0: передали прямо файл history.csv
    if path.is_file():
        if path.name != "history.csv":
            raise ValueError(f"Expected history.csv file, got: {path}")
        path = path.parent

    # Case 1: history.csv лежит прямо в path
    if (path / "history.csv").exists():
        df_history, df_config = load_direct_history_dir(path)

        print("mode: direct history.csv")
        print("path:", path)
        print("runs loaded:", df_history["run_name"].nunique())
        print("history rows:", len(df_history))

        return df_history, df_config

    # Case 2: path сам является experiment_dir с runs/
    if (path / "runs").is_dir():
        df_history, df_config = load_experiment_dir(path)

        print("mode: experiment dir")
        print("experiment:", path.name)
        print("runs loaded:", df_history["run_name"].nunique())
        print("history rows:", len(df_history))

        return df_history, df_config

    # Case 3: path является studies_root
    exp_dir = get_last_exp(path)
    df_history, df_config = load_experiment_dir(exp_dir)

    print("mode: studies root -> last experiment")
    print("studies_root:", path)
    print("experiment:", exp_dir.name)
    print("runs loaded:", df_history["run_name"].nunique())
    print("history rows:", len(df_history))

    return df_history, df_config


def perform_last_exp(
    studies_root: Path = STUDIES_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp_dir = get_last_exp(studies_root)
    return perform_exp(exp_dir)


# =========================
# SUMMARY
# =========================

def make_summary(
    df_history: pd.DataFrame,
    metric_col: str = "valid_accuracy",
) -> pd.DataFrame:
    if metric_col not in df_history.columns:
        raise ValueError(f"Column '{metric_col}' not found in df_history")

    rows = []

    for run_name, g in df_history.groupby("run_name", sort=False):
        g = g.copy()

        best_row = g.loc[g[metric_col].idxmax()]
        final_row = g.iloc[-1]

        rows.append({
            "run_name": run_name,
            "run_label": final_row.get("run_label"),
            "config_lambda_coef": final_row.get("config_lambda_coef"),
            "first_lambda_coef": (
                float(g["lambda_coef"].dropna().iloc[0])
                if "lambda_coef" in g.columns and len(g["lambda_coef"].dropna()) > 0
                else None
            ),
            "final_lambda_coef": (
                float(g["lambda_coef"].dropna().iloc[-1])
                if "lambda_coef" in g.columns and len(g["lambda_coef"].dropna()) > 0
                else None
            ),
            "best_epoch": int(best_row["epoch"]) if "epoch" in g.columns else None,
            f"best_{metric_col}": float(best_row[metric_col]),
            "final_epoch": int(final_row["epoch"]) if "epoch" in g.columns else None,
            f"final_{metric_col}": float(final_row[metric_col]),
            "num_epochs": int(g["epoch"].nunique()) if "epoch" in g.columns else len(g),
            "num_rows": len(g),
            "run_dir": final_row.get("run_dir"),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(f"best_{metric_col}", ascending=False)
        .reset_index(drop=True)
    )


# =========================
# PLOTTING
# =========================

def plot_metric_by_epoch(
    df_history: pd.DataFrame,
    metric: str,
    title: str | None = None,
    ylabel: str | None = None,
    average_by_lambda: bool = False,
    run_number: int | None = None,
    run_name: str | None = None,
    label_col: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    figsize: tuple[int, int] = (12, 7),
):
    if metric not in df_history.columns:
        raise ValueError(f"Column '{metric}' not found in df_history")

    if "epoch" not in df_history.columns:
        raise ValueError("Column 'epoch' not found in df_history")

    if xscale not in {"linear", "log"}:
        raise ValueError("xscale must be 'linear' or 'log'")

    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'")

    df = df_history.copy()

    if "run_name" not in df.columns:
        df["run_name"] = "single_run"

    if label_col is None:
        if "run_label" in df.columns:
            label_col = "run_label"
        elif "lambda_str" in df.columns:
            label_col = "lambda_str"
        else:
            label_col = "run_name"

    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found in df_history")

    run_names = sorted(df["run_name"].dropna().unique())

    if run_name is not None and run_number is not None:
        raise ValueError("Pass either run_name or run_number, not both")

    if run_number is not None:
        if run_number < 1 or run_number > len(run_names):
            raise ValueError(f"run_number must be from 1 to {len(run_names)}")
        run_name = run_names[run_number - 1]

    if run_name is not None:
        df = df[df["run_name"] == run_name].copy()

    df = df.dropna(subset=["epoch", metric])

    if xscale == "log":
        df = df[df["epoch"] > 0]

    if yscale == "log":
        df = df[df[metric] > 0]

    fig, ax = plt.subplots(figsize=figsize)

    if average_by_lambda:
        group_col = "config_lambda_coef" if "config_lambda_coef" in df.columns else label_col

        plot_df = (
            df
            .dropna(subset=[group_col, metric])
            .groupby([group_col, "epoch"], as_index=False)
            .agg(
                metric_mean=(metric, "mean"),
                metric_std=(metric, "std"),
                num_runs=(metric, "count"),
            )
            .sort_values([group_col, "epoch"])
        )

        for group_value, g in plot_df.groupby(group_col, sort=False):
            g = g.sort_values("epoch")

            ax.plot(
                g["epoch"],
                g["metric_mean"],
                linewidth=2.2,
                label=str(group_value),
            )

    else:
        # Группируем по run_name, а не по lambda_coef.
        # Иначе adaptive lambda разрезает одну траекторию на много кусков.
        for run_name_i, g in df.groupby("run_name", sort=False, dropna=False):
            g = g.sort_values("epoch")

            label = str(g[label_col].iloc[0])

            ax.plot(
                g["epoch"],
                g[metric],
                alpha=0.75,
                linewidth=1.6,
                label=label,
            )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        title=label_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel or metric)

    plot_title = title or f"{metric} / epoch"
    if run_name is not None:
        plot_title += f"\nrun: {run_name}"

    ax.set_title(plot_title)

    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)

    plt.tight_layout()
    plt.show()


def print_available_metrics(df_history: pd.DataFrame) -> None:
    useful_keywords = [
        "accuracy",
        "loss",
        "lambda",
        "zero",
        "active",
        "real_prob",
        "estim_prob",
    ]

    cols = [
        c for c in df_history.columns
        if any(k in c for k in useful_keywords)
    ]

    for c in cols:
        print(c)