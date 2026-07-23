from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


try:
    import yaml
except ImportError:
    yaml = None


STUDIES_ROOT = Path("PUT_YOUR_STUDIES_ROOT_HERE")

CONFIG_NAMES = [
    "config_resolved.yaml",
    "configs_resolved.yaml",
    "resolved_config.yaml",
    "config.yaml",
    "configs.yaml",
]

# Backward-compatible alias for older imports.
CONFIG_NAME = CONFIG_NAMES[0]


ACC_CANDIDATES = [
    "valid_accuracy",
    "val_accuracy",
    "valid_acc",
    "val_acc",
    "accuracy",
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


def _get_nested(d: dict, keys: list[str], default=None):
    cur = d

    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]

    return cur


def find_config_path(run_dir: Path) -> Path | None:
    run_dir = Path(run_dir)

    for name in CONFIG_NAMES:
        path = run_dir / name
        if path.exists():
            return path

    return None


def read_config(config_path: Path | None) -> dict:
    if yaml is None or config_path is None:
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _read_yaml(path: Path) -> dict:
    if yaml is None or not path.exists():
        return {}

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def extract_lambda_from_config(config: dict):
    """
    Поддерживает два случая:
    1. Обычный запуск: model.lambda_coef
    2. Warmup/adaptive lambda: training_arguments.lambda_warmup.target_lambda_coef
    """

    warmup_cfg = _get_nested(config, ["training_arguments", "lambda_warmup"], {})
    if isinstance(warmup_cfg, dict):
        enabled = warmup_cfg.get("enabled", False)
        target = warmup_cfg.get("target_lambda_coef", None)

        if enabled and target is not None:
            return target

    value = _get_nested(config, ["model", "lambda_coef"], None)
    if value is not None:
        return value

    value = _get_nested(config, ["training_arguments", "lambda_coef"], None)
    if value is not None:
        return value

    return np.nan


def get_lambda_coef_from_config(run_dir: Path):
    config_path = find_config_path(run_dir)
    if config_path is None:
        return np.nan

    config = _read_yaml(config_path)
    return extract_lambda_from_config(config)


def get_config_lambda_coef(cfg: dict) -> float | None:
    value = extract_lambda_from_config(cfg)

    if pd.isna(value):
        return None

    return float(value)


# backward-compatible alias
def get_lambda_coef(cfg: dict) -> float | None:
    return get_config_lambda_coef(cfg)


def _find_acc_col(df: pd.DataFrame) -> str | None:
    for col in ACC_CANDIDATES:
        if col in df.columns:
            return col

    for col in df.columns:
        low = col.lower()
        if "valid" in low and ("acc" in low or "accuracy" in low):
            return col

    return None


# =========================
# HISTORY NORMALIZATION
# =========================

def _format_float(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "unknown"
    return f"{float(x):g}"


def _format_label_value(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if value != 0.0 and abs(value) <= 1e-3:
            mantissa, exponent = f"{value:.6e}".split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            return f"{mantissa}e{int(exponent)}"
        formatted = f"{value:g}"
        return re.sub(r"e([+-])0+(\d+)$", r"e\1\2", formatted)
    return str(value).replace(" ", "_")


def extract_run_label_values(config: dict) -> dict[str, object]:
    """Extract aliased run-label values declared in reporting.run_label_fields."""
    fields = _get_nested(config, ["reporting", "run_label_fields"], {})
    if fields is None:
        return {}
    if not isinstance(fields, dict):
        raise ValueError("reporting.run_label_fields must be a mapping of alias to config path")

    values = {}
    for alias, path in fields.items():
        if not isinstance(path, str) or not path:
            raise ValueError(
                f"reporting.run_label_fields.{alias} must be a non-empty dotted path"
            )
        value = _get_nested(config, path.split("."), None)
        if value is None:
            raise ValueError(f"Run-label config path not found: {path}")
        if isinstance(value, (dict, list, tuple)):
            raise ValueError(f"Run-label config path must point to a scalar: {path}")
        values[str(alias)] = value
    return values


def make_run_label(
    history: pd.DataFrame,
    run_name: str,
    config_lambda_coef: float | None,
    label_values: dict[str, object] | None = None,
) -> str:
    """
    Статический label для легенды.

    Важно:
    - lambda_coef в history может меняться по эпохам;
    - поэтому нельзя использовать per-row lambda_coef как lambda_str для groupby.
    """
    if label_values:
        return "_".join(
            f"{alias}_{_format_label_value(value)}"
            for alias, value in label_values.items()
        )

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


def _add_zero_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для Gumbel-логов:
    valid_*zero_prob* — вероятность зануления канала.

    Добавляет:
    - zero_channels: сколько каналов считаются закрытыми
    - open_channels: сколько каналов считаются открытыми
    - expected_open_channels: сумма вероятностей открытия
    """

    df = df.copy()

    zero_cols = [
        c for c in df.columns
        if "zero_prob" in c.lower() and c.startswith("valid")
    ]

    if len(zero_cols) == 0:
        return df

    zero_probs = df[zero_cols].apply(pd.to_numeric, errors="coerce")

    df["zero_channels"] = (zero_probs >= 0.5).sum(axis=1)
    df["open_channels"] = (zero_probs < 0.5).sum(axis=1)
    df["expected_open_channels"] = (1.0 - zero_probs).sum(axis=1)
    df["mean_zero_prob"] = zero_probs.mean(axis=1)

    return df


def _add_aig_gate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для AIG-логов:
    valid_g_prob_* — средняя вероятность/частота открытия блока.
    """

    df = df.copy()

    gate_cols = [
        c for c in df.columns
        if c.startswith("valid_g_prob_")
    ]

    if len(gate_cols) == 0:
        return df

    gates = df[gate_cols].apply(pd.to_numeric, errors="coerce")

    df["valid_active_blocks"] = (gates >= 0.5).sum(axis=1)
    df["valid_inactive_blocks"] = (gates < 0.5).sum(axis=1)
    df["valid_active_blocks_expected"] = gates.sum(axis=1)
    df["valid_inactive_blocks_expected"] = len(gate_cols) - gates.sum(axis=1)
    df["valid_mean_gate_prob"] = gates.mean(axis=1)

    return df


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
    else:
        history["epoch"] = np.arange(len(history))

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

    history = _add_zero_prob_columns(history)
    history = _add_aig_gate_columns(history)

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
    label_values = extract_run_label_values(cfg)

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
        label_values=label_values,
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
        **{f"label_{alias}": value for alias, value in label_values.items()},
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


def load_history(run_dir: Path) -> pd.DataFrame:
    run_dir = Path(run_dir)

    history, _ = load_single_run(
        run_dir=run_dir,
        experiment_name=run_dir.parent.name,
        experiment_dir=run_dir.parent,
        run_name=run_dir.name,
    )

    return history


def summarize_one_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    df = load_history(run_dir)

    acc_col = _find_acc_col(df)

    result = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "config_lambda_coef": df["config_lambda_coef"].iloc[0],
        "lambda_str": df["lambda_str"].iloc[0],
        "num_epochs": len(df),
        "final_epoch": df["epoch"].iloc[-1],
    }

    if acc_col is not None:
        best_idx = df[acc_col].idxmax()
        best_row = df.loc[best_idx]
        final_row = df.iloc[-1]

        result.update({
            "acc_col": acc_col,
            "best_epoch": best_row["epoch"],
            "best_valid_accuracy": best_row[acc_col],
            "final_valid_accuracy": final_row[acc_col],
        })

    for col in [
        "zero_channels",
        "open_channels",
        "expected_open_channels",
        "mean_zero_prob",
        "valid_active_blocks_expected",
        "valid_inactive_blocks_expected",
        "valid_mean_gate_prob",
        "valid_active_channels",
        "valid_zero_channels",
        "valid_active_ratio",
        "valid_zero_ratio",
        "valid_aig_static_flops_per_sample",
        "valid_aig_active_flops_per_sample",
        "valid_aig_skipped_flops_per_sample",
        "valid_aig_flops_skip_ratio",
        "valid_aig_flops_active_ratio",
        "valid_aig_gated_branch_flops_per_sample",
        "lambda_coef",
    ]:
        if col in df.columns:
            result[f"final_{col}"] = df[col].iloc[-1]

            if acc_col is not None:
                best_idx = df[acc_col].idxmax()
                result[f"best_{col}"] = df.loc[best_idx, col]

    return result


def _find_run_dirs(study_dir: Path) -> list[Path]:
    study_dir = Path(study_dir)

    if (study_dir / "runs").exists():
        root = study_dir / "runs"
    else:
        root = study_dir

    run_dirs = [
        p for p in root.iterdir()
        if p.is_dir() and (p / "history.csv").exists()
    ]

    return sorted(run_dirs)


def collect_runs(study_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Возвращает:
    - summary_df: одна строка на запуск
    - history_df: все эпохи всех запусков
    """

    run_dirs = _find_run_dirs(Path(study_dir))

    summaries = []
    histories = []

    for run_dir in run_dirs:
        try:
            history, config_row = load_single_run(
                run_dir=run_dir,
                experiment_name=Path(study_dir).name,
                experiment_dir=Path(study_dir),
                run_name=run_dir.name,
            )
            summary = summarize_one_run(run_dir)
            summary.update(config_row)
        except Exception as e:
            print(f"[skip] {run_dir}: {e}")
            continue

        histories.append(history)
        summaries.append(summary)

    history_df = (
        pd.concat(histories, ignore_index=True)
        if histories
        else pd.DataFrame()
    )

    summary_df = pd.DataFrame(summaries)

    if not summary_df.empty:
        summary_df, history_df = _apply_automatic_run_labels(summary_df, history_df)
        if "best_valid_accuracy" in summary_df.columns:
            summary_df = summary_df.sort_values(
                "best_valid_accuracy",
                ascending=False,
            )
        else:
            summary_df = summary_df.sort_values("run_name")

    return summary_df, history_df


_AUTO_RUN_LABEL_FIELDS = (
    ("lambda_init", "model.lambda_coef"),
    ("entropy", "model.entropy_regularization"),
    ("regularization", "model.backbone.resnet_block.gate_regularization"),
    ("step", "training_arguments.adaptive_lambda.log_step_init"),
    ("update", "training_arguments.adaptive_lambda.update_every_epochs"),
)


def _apply_automatic_run_labels(
    summary_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build concise labels from config fields that vary inside the study."""
    explicit_fields = [
        col for col in summary_df.columns if col.startswith("reporting.run_label_fields.")
    ]
    if explicit_fields or summary_df.empty or history_df.empty:
        return summary_df, history_df

    varying_fields = [
        (alias, path)
        for alias, path in _AUTO_RUN_LABEL_FIELDS
        if path in summary_df.columns and summary_df[path].dropna().nunique() > 1
    ]
    if not varying_fields:
        return summary_df, history_df

    summary_df = summary_df.copy()
    summary_df["run_label"] = summary_df.apply(
        lambda row: "_".join(
            f"{alias}_{_format_label_value(row[path])}"
            for alias, path in varying_fields
        ),
        axis=1,
    )
    label_by_run = summary_df.set_index("run_name")["run_label"]

    history_df = history_df.copy()
    mapped_labels = history_df["run_name"].map(label_by_run)
    history_df["run_label"] = mapped_labels.fillna(history_df["run_label"])
    history_df["lambda_str"] = history_df["run_label"]
    return summary_df, history_df


def load_study(study_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load either one run directory or a study containing several runs."""
    study_dir = Path(study_dir)
    if (study_dir / "history.csv").exists():
        history_df = load_history(study_dir)
        summary_df = pd.DataFrame([summarize_one_run(study_dir)])
        return summary_df, history_df
    return collect_runs(study_dir)


# =========================
# SUMMARY
# =========================

def make_summary(
    df_history: pd.DataFrame,
    metric_col: str = "valid_accuracy",
) -> pd.DataFrame:
    if metric_col not in df_history.columns:
        acc_col = _find_acc_col(df_history)
        if metric_col == "valid_accuracy" and acc_col is not None:
            metric_col = acc_col
        else:
            raise ValueError(f"Column '{metric_col}' not found in df_history")

    rows = []

    for run_name, g in df_history.groupby("run_name", sort=False):
        g = g.copy()

        best_row = g.loc[g[metric_col].idxmax()]
        final_row = g.iloc[-1]

        row = {
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
        }

        for col in [
            "zero_channels",
            "open_channels",
            "expected_open_channels",
            "mean_zero_prob",
            "valid_active_blocks_expected",
            "valid_inactive_blocks_expected",
            "valid_mean_gate_prob",
            "valid_active_channels",
            "valid_zero_channels",
            "valid_active_ratio",
            "valid_zero_ratio",
            "valid_aig_static_flops_per_sample",
            "valid_aig_active_flops_per_sample",
            "valid_aig_skipped_flops_per_sample",
            "valid_aig_flops_skip_ratio",
            "valid_aig_flops_active_ratio",
            "valid_aig_gated_branch_flops_per_sample",
        ]:
            if col in g.columns:
                row[f"best_{col}"] = best_row[col]
                row[f"final_{col}"] = final_row[col]

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(f"best_{metric_col}", ascending=False)
        .reset_index(drop=True)
    )


# =========================
# PLOTTING
# =========================

def _plot_metric_interactive(
    history_df: pd.DataFrame,
    metric: str,
    *,
    x_col: str,
    run_col: str,
    label_col: str | None,
    xscale: str,
    yscale: str,
    title: str | None,
    ylabel: str | None,
    show_mean: bool,
    average_by: str | None,
    run_name: str | None,
    show: bool,
):
    plot_df = history_df.copy()
    if run_col not in plot_df.columns:
        plot_df[run_col] = "single_run"
    if run_name is not None:
        plot_df = plot_df[plot_df[run_col] == run_name].copy()
        if plot_df.empty:
            raise ValueError(f"No run '{run_name}' in column '{run_col}'")
    if label_col is None:
        label_col = next(
            (col for col in ("run_label", "lambda_str", run_col) if col in plot_df.columns),
            run_col,
        )
    if label_col not in plot_df.columns:
        raise ValueError(f"label_col='{label_col}' not found in history_df")

    plot_df = plot_df.dropna(subset=[x_col, metric])
    if xscale == "log":
        plot_df = plot_df[plot_df[x_col] > 0]
    if yscale == "log":
        plot_df = plot_df[plot_df[metric] > 0]

    figure = go.Figure()
    if average_by is not None:
        if average_by not in plot_df.columns:
            raise ValueError(f"average_by='{average_by}' not found in history_df")
        grouped = (
            plot_df.groupby([average_by, x_col], as_index=False)[metric]
            .mean()
            .sort_values([average_by, x_col])
        )
        for group_value, group in grouped.groupby(average_by, sort=False):
            label = str(group_value)
            figure.add_trace(
                go.Scatter(
                    x=group[x_col],
                    y=group[metric],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                )
            )
    else:
        seen_labels = set()
        for _, group in plot_df.groupby(run_col, sort=False):
            group = group.sort_values(x_col)
            label = str(group[label_col].iloc[0])
            figure.add_trace(
                go.Scatter(
                    x=group[x_col],
                    y=group[metric],
                    mode="lines",
                    name=label,
                    legendgroup=label,
                    showlegend=label not in seen_labels,
                    hovertemplate=(
                        f"{label}<br>{x_col}=%{{x}}<br>{metric}=%{{y:.6g}}<extra></extra>"
                    ),
                )
            )
            seen_labels.add(label)

        if show_mean:
            for label, group in plot_df.groupby(label_col, sort=False):
                mean_df = group.groupby(x_col, as_index=False)[metric].mean()
                figure.add_trace(
                    go.Scatter(
                        x=mean_df[x_col],
                        y=mean_df[metric],
                        mode="lines",
                        line={"width": 4, "dash": "dash"},
                        name=f"mean {label}",
                        legendgroup=str(label),
                    )
                )

    figure.update_layout(
        title=title or f"{metric} / {x_col}",
        xaxis_title=x_col,
        yaxis_title=ylabel or metric,
        xaxis_type=xscale,
        yaxis_type=yscale,
        hovermode="x unified",
        legend={
            "x": 1.02,
            "y": 1.0,
            "xanchor": "left",
            "yanchor": "top",
            "groupclick": "togglegroup",
        },
        margin={"r": 320},
        template="plotly_white",
    )
    if show:
        figure.show()
    return figure

def plot_metric(
    history_df: pd.DataFrame,
    metric: str,
    *,
    x_col: str = "epoch",
    run_col: str = "run_name",
    label_col: str | None = None,
    yscale: str = "linear",
    xscale: str = "linear",
    title: str | None = None,
    ylabel: str | None = None,
    figsize: tuple = (10, 5),
    alpha: float = 0.75,
    linewidth: float = 1.6,
    show_legend: bool = True,
    show_mean: bool = False,
    average_by: str | None = None,
    run_name: str | None = None,
    ax=None,
    show: bool = True,
    interactive: bool = False,
):
    """Plot one history metric with optional run filtering and averaging."""

    if history_df.empty:
        raise ValueError("history_df is empty")

    if metric not in history_df.columns:
        raise ValueError(
            f"No column '{metric}' in history_df. "
            f"Available columns: {list(history_df.columns)}"
        )

    if x_col not in history_df.columns:
        raise ValueError(f"No x_col '{x_col}' in history_df")

    if xscale not in {"linear", "log"}:
        raise ValueError("xscale must be 'linear' or 'log'")
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'")

    if interactive:
        return _plot_metric_interactive(
            history_df,
            metric,
            x_col=x_col,
            run_col=run_col,
            label_col=label_col,
            xscale=xscale,
            yscale=yscale,
            title=title,
            ylabel=ylabel,
            show_mean=show_mean,
            average_by=average_by,
            run_name=run_name,
            show=show,
        )

    plot_df = history_df.copy()
    if run_col not in plot_df.columns:
        plot_df[run_col] = "single_run"

    if run_name is not None:
        plot_df = plot_df[plot_df[run_col] == run_name].copy()
        if plot_df.empty:
            raise ValueError(f"No run '{run_name}' in column '{run_col}'")

    if label_col is None:
        label_col = next(
            (col for col in ("run_label", "lambda_str", run_col) if col in plot_df.columns),
            run_col,
        )
    if label_col not in plot_df.columns:
        raise ValueError(f"label_col='{label_col}' not found in history_df")

    plot_df = plot_df.dropna(subset=[x_col, metric])
    if xscale == "log":
        plot_df = plot_df[plot_df[x_col] > 0]
    if yscale == "log":
        plot_df = plot_df[plot_df[metric] > 0]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if average_by is not None:
        if average_by not in plot_df.columns:
            raise ValueError(f"average_by='{average_by}' not found in history_df")

        grouped = (
            plot_df.groupby([average_by, x_col], as_index=False)[metric]
            .mean()
            .sort_values([average_by, x_col])
        )
        for group_value, group in grouped.groupby(average_by, sort=False):
            ax.plot(
                group[x_col],
                group[metric],
                linewidth=linewidth,
                label=str(group_value),
            )
    else:
        plot_df = plot_df.sort_values([run_col, x_col])

        for run_value, group in plot_df.groupby(run_col, sort=False):
            label = str(group[label_col].iloc[0])
            ax.plot(
                group[x_col],
                group[metric],
                alpha=alpha,
                linewidth=linewidth,
                label=label if label else str(run_value),
            )

    if show_mean and average_by is None:
        for label_value, g in plot_df.groupby(label_col):
            mean_df = (
                g.groupby(x_col, as_index=False)[metric]
                .mean()
                .dropna(subset=[metric])
            )

            ax.plot(
                mean_df[x_col],
                mean_df[metric],
                linewidth=3.0,
                linestyle="--",
                label=f"mean {label_value}",
            )

    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} / {x_col}")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)

    if show_legend and ax.lines:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)

    ax.figure.tight_layout()
    if show:
        plt.show()
    return ax


def plot_acc_epoch(
    history_df: pd.DataFrame,
    *,
    acc_col: str | None = None,
    yscale: str = "linear",
    show_mean: bool = False,
    interactive: bool = False,
):
    if acc_col is None:
        acc_col = _find_acc_col(history_df)

    if acc_col is None:
        raise ValueError("Cannot find accuracy column")

    return plot_metric(
        history_df,
        acc_col,
        yscale=yscale,
        title=f"{acc_col} / epoch",
        show_mean=show_mean,
        interactive=interactive,
    )


def plot_channel_counts(
    history_df: pd.DataFrame,
    *,
    active_col: str | None = None,
    closed_col: str | None = None,
    total_channels: float | None = None,
    run_name: str | None = None,
    figsize: tuple = (10, 5),
    show: bool = True,
    interactive: bool = False,
):
    """Plot thresholded open channel or block counts with count and percentage axes."""
    if (active_col is None) != (closed_col is None):
        raise ValueError("Pass active_col and closed_col together")

    unit = "channels"
    if active_col is None:
        aig_counts = (
            "valid_active_blocks" in history_df.columns
            and "valid_inactive_blocks" in history_df.columns
        )
        candidates = (
            (
                ("valid_active_blocks", "valid_inactive_blocks", "blocks"),
                ("valid_real_active_channels", "valid_real_zero_channels", "channels"),
                ("open_channels", "zero_channels", "channels"),
            )
            if aig_counts
            else (
                ("valid_real_active_channels", "valid_real_zero_channels", "channels"),
                ("open_channels", "zero_channels", "channels"),
            )
        )
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate[0] in history_df.columns and candidate[1] in history_df.columns
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                "Cannot derive factual open/closed counts. Expected real count "
                "columns, per-channel valid_*zero_prob* columns, or AIG "
                "valid_g_prob_* columns in history.csv."
            )
        active_col, closed_col, unit = selected

    required = {"epoch", active_col, closed_col}
    missing = sorted(required - set(history_df.columns))
    if missing:
        raise ValueError(
            f"Count columns not found: {missing}."
        )

    plot_df = history_df.copy()
    if "run_name" not in plot_df.columns:
        plot_df["run_name"] = "single_run"
    if "run_label" not in plot_df.columns:
        plot_df["run_label"] = plot_df["run_name"]
    if run_name is not None:
        plot_df = plot_df[plot_df["run_name"] == run_name].copy()
        if plot_df.empty:
            raise ValueError(f"No run '{run_name}' in history_df")

    plot_df = plot_df.dropna(subset=["epoch", active_col, closed_col])
    inferred_totals = plot_df[active_col] + plot_df[closed_col]
    if total_channels is None:
        if inferred_totals.empty:
            raise ValueError("No channel-count values to plot")
        total_channels = float(inferred_totals.median())
        if not np.allclose(inferred_totals, total_channels, rtol=1e-5, atol=1e-5):
            raise ValueError(
                "The inferred total number of channels is not constant. "
                "Pass total_channels explicitly or select one compatible run."
            )
    if total_channels <= 0:
        raise ValueError("total_channels must be positive")

    if interactive:
        figure = go.Figure()
        colors = (
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
        )
        seen_labels = set()
        color_by_label = {}
        for _, group in plot_df.groupby("run_name", sort=False):
            group = group.sort_values("epoch")
            label = str(group["run_label"].iloc[0])
            if label not in color_by_label:
                color_by_label[label] = colors[len(color_by_label) % len(colors)]
            color = color_by_label[label]
            figure.add_trace(
                go.Scatter(
                    x=group["epoch"],
                    y=group[active_col],
                    mode="lines",
                    line={"color": color, "width": 2},
                    name=label,
                    legendgroup=label,
                    showlegend=label not in seen_labels,
                    hovertemplate=(
                        f"{label}<br>state=open<br>epoch=%{{x}}<br>{unit}=%{{y:.4g}}"
                        "<extra></extra>"
                    ),
                )
            )
            seen_labels.add(label)

        max_count = float(plot_df[active_col].max())
        percent_max = 100.0 * max_count * 1.05 / total_channels
        figure.add_trace(
            go.Scatter(
                x=[plot_df["epoch"].min(), plot_df["epoch"].max()],
                y=[0.0, percent_max],
                yaxis="y2",
                opacity=0.0,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.update_layout(
            title=f"Factual open {unit}",
            xaxis_title="epoch",
            yaxis={"title": unit, "rangemode": "tozero"},
            yaxis2={
                "title": f"{unit}, % of total",
                "overlaying": "y",
                "side": "right",
                "range": [0.0, percent_max],
            },
            hovermode="x unified",
            legend={
                "x": 1.08,
                "y": 1.0,
                "xanchor": "left",
                "yanchor": "top",
                "groupclick": "togglegroup",
                "title": {"text": "Click runs to show/hide"},
            },
            margin={"r": 380},
            template="plotly_white",
        )
        if show:
            figure.show()
        return figure

    _, ax = plt.subplots(figsize=figsize)
    for current_run, group in plot_df.groupby("run_name", sort=False):
        group = group.sort_values("epoch")
        label = str(group["run_label"].iloc[0] or current_run)
        ax.plot(
            group["epoch"],
            group[active_col],
            label=label,
            linewidth=1.8,
        )

    percent_axis = ax.secondary_yaxis(
        "right",
        functions=(
            lambda count: 100.0 * count / total_channels,
            lambda percent: percent * total_channels / 100.0,
        ),
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel(unit)
    percent_axis.set_ylabel(f"{unit}, % of total")
    ax.set_title(f"Factual open {unit}")
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    ax.legend(fontsize=8)
    ax.figure.tight_layout()
    if show:
        plt.show()
    return ax


def plot_lambda_epoch(
    history_df: pd.DataFrame,
    *,
    lambda_col: str = "lambda_coef",
    **kwargs,
):
    """Plot lambda by epoch on a logarithmic y-axis."""
    return plot_metric(
        history_df,
        lambda_col,
        yscale="log",
        title=f"{lambda_col} / epoch",
        **kwargs,
    )


def plot_core_statistics(
    history_df: pd.DataFrame,
    *,
    run_name: str | None = None,
    interactive: bool = True,
) -> dict[str, object]:
    """Plot the standard accuracy, factual channel-count, and lambda charts."""
    view_df = history_df
    if run_name is not None:
        view_df = history_df[history_df["run_name"] == run_name].copy()
        if view_df.empty:
            raise ValueError(f"No run '{run_name}' in history_df")
    return {
        "accuracy": plot_acc_epoch(view_df, interactive=interactive),
        "channels": plot_channel_counts(view_df, interactive=interactive),
        "lambda": plot_lambda_epoch(view_df, interactive=interactive),
    }


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
    if run_name is not None and run_number is not None:
        raise ValueError("Pass either run_name or run_number, not both")

    if run_number is not None:
        if "run_name" not in df_history.columns:
            run_names = ["single_run"]
        else:
            run_names = sorted(df_history["run_name"].dropna().unique())
        if run_number < 1 or run_number > len(run_names):
            raise ValueError(f"run_number must be from 1 to {len(run_names)}")
        run_name = run_names[run_number - 1]

    average_by = None
    if average_by_lambda:
        average_by = next(
            (
                col
                for col in ("config_lambda_coef", label_col, "lambda_str", "run_label")
                if col is not None and col in df_history.columns
            ),
            None,
        )
        if average_by is None:
            raise ValueError("Cannot find a lambda column for averaging")

    return plot_metric(
        df_history,
        metric,
        title=title,
        ylabel=ylabel,
        label_col=label_col,
        xscale=xscale,
        yscale=yscale,
        figsize=figsize,
        average_by=average_by,
        run_name=run_name,
    )


def plot_metrics(
    history_df: pd.DataFrame,
    metrics: list[str] | tuple[str, ...],
    **kwargs,
) -> dict[str, object]:
    """Plot several metrics using the same options and return their axes."""
    return {metric: plot_metric(history_df, metric, **kwargs) for metric in metrics}


def plot_gradient_norms(
    history_df: pd.DataFrame,
    *,
    components: tuple[str, ...] = ("ce", "regularization", "total"),
    parameter_group: str = "total",
    statistic: str = "mean",
    **kwargs,
) -> dict[str, object]:
    """Plot the standard component-wise gradient-norm metrics."""
    metrics = [
        f"grad_norm_{component}_{parameter_group}_{statistic}"
        for component in components
    ]
    missing = [metric for metric in metrics if metric not in history_df.columns]
    if missing:
        available = sorted(
            col for col in history_df.columns if col.startswith("grad_norm_")
        )
        raise ValueError(
            f"Gradient norm columns not found: {missing}. Available: {available}"
        )
    return plot_metrics(history_df, metrics, **kwargs)


_GRADIENT_NORM_RE = re.compile(
    r"^grad_norm_(ce|regularization|total)_(.+)_(mean|max)$"
)


def gradient_norm_catalog(history_df: pd.DataFrame) -> pd.DataFrame:
    """Describe available gradient-norm columns in a notebook-friendly table."""
    rows = []
    for metric in history_df.columns:
        match = _GRADIENT_NORM_RE.match(metric)
        if match is None:
            continue
        component, parameter_group, statistic = match.groups()
        rows.append(
            {
                "metric": metric,
                "component": component,
                "parameter_group": parameter_group,
                "statistic": statistic,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["metric", "component", "parameter_group", "statistic"],
    ).sort_values(["parameter_group", "component", "statistic"], ignore_index=True)


def print_available_metrics(df_history: pd.DataFrame) -> None:
    useful_keywords = [
        "accuracy",
        "loss",
        "lambda",
        "zero",
        "active",
        "flops",
        "compute",
        "real_prob",
        "estim_prob",
        "grad_norm",
    ]

    cols = [
        c for c in df_history.columns
        if any(k in c for k in useful_keywords)
    ]

    for c in cols:
        print(c)
