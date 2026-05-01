from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import optuna
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from net_complexity.training.engine import run_training
from net_complexity.tuning.flags import install_tune_cli_flags
from net_complexity.tuning.restart_guard import RepeatRestartGuard, RepeatRestartRequested
from net_complexity.tuning.repeats import (
    resolve_repeat_attempt_seed,
    resolve_repeat_seeds,
    select_best_repeat,
)
from net_complexity.tuning.search import (
    GRID_POINT_INDEX_PARAM,
    build_grid_points,
    build_grid_search_space,
    build_point_grid_search_space,
    count_grid_trials,
)


CONFIGS_PATH = str(Path(__file__).resolve().parents[3] / "configs")


CLI_TUNING_OVERRIDES: dict[str, Any] = {}
LAMBDA_CONFIG_PATH = "model.lambda_coef"


def initialize_cli_flags() -> None:
    global CLI_TUNING_OVERRIDES
    CLI_TUNING_OVERRIDES = install_tune_cli_flags()


def _clone_config(config: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def _format_tuning_progress(
    *,
    display_trial_idx: int,
    trial_total: int,
    optuna_trial_number: int,
    repeat_number: int | None = None,
    repeat_total: int | None = None,
    attempt_number: int | None = None,
    attempt_total: int | None = None,
) -> str:
    parts = [
        f"display_trial={display_trial_idx}/{trial_total}",
        f"optuna_trial_number={optuna_trial_number}",
    ]
    if repeat_number is not None and repeat_total is not None:
        parts.append(f"repeat={repeat_number}/{repeat_total}")
    if attempt_number is not None and attempt_total is not None:
        parts.append(f"attempt={attempt_number}/{attempt_total}")
    return " | ".join(parts)


def _assert_lambda_matches_config(
    config: DictConfig,
    lambda_value: Any,
    *,
    source: str,
) -> None:
    cfg_lambda = OmegaConf.select(config, LAMBDA_CONFIG_PATH)
    assert cfg_lambda is not None, f"{LAMBDA_CONFIG_PATH} is missing in config while checking {source}."
    assert abs(float(cfg_lambda) - float(lambda_value)) < 1e-12, (
        f"{source}={lambda_value} does not match cfg.{LAMBDA_CONFIG_PATH}={cfg_lambda}."
    )


def _suggest_value(trial: optuna.Trial, name: str, spec: DictConfig) -> Any:
    suggestion_type = str(spec.type).lower()
    if suggestion_type == "float":
        step = getattr(spec, "step", None)
        return trial.suggest_float(
            name,
            float(spec.low),
            float(spec.high),
            step=float(step) if step is not None else None,
            log=bool(getattr(spec, "log", False)),
        )
    if suggestion_type == "int":
        step = int(getattr(spec, "step", 1))
        return trial.suggest_int(
            name,
            int(spec.low),
            int(spec.high),
            step=step,
            log=bool(getattr(spec, "log", False)),
        )
    if suggestion_type == "categorical":
        return trial.suggest_categorical(name, list(spec.choices))
    raise ValueError(f"Unsupported search space type '{spec.type}' for '{name}'.")


def _apply_optuna_search_space(
    trial: optuna.Trial,
    config: DictConfig,
    search_space: DictConfig,
) -> dict[str, Any]:
    suggested_params: dict[str, Any] = {}
    for path, spec in search_space.items():
        value = _suggest_value(trial, path, spec)
        OmegaConf.update(config, path, value, merge=False)
        suggested_params[path] = value
    return suggested_params


def _apply_grid_search_space(
    trial: optuna.Trial,
    config: DictConfig,
    grid_search_space: dict[str, list[Any]],
) -> dict[str, Any]:
    suggested_params: dict[str, Any] = {}
    for path, values in grid_search_space.items():
        value = trial.suggest_categorical(path, list(values))
        OmegaConf.update(config, path, value, merge=False)
        suggested_params[path] = value
    return suggested_params


def _apply_grid_points(
    trial: optuna.Trial,
    config: DictConfig,
    grid_points: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    point_index = int(
        trial.suggest_categorical(GRID_POINT_INDEX_PARAM, list(range(len(grid_points))))
    )
    suggested_params = dict(grid_points[point_index])
    for path, value in suggested_params.items():
        OmegaConf.update(config, path, value, merge=False)
    return point_index, suggested_params


def _update_trial_metadata(config: DictConfig, tuning_cfg: DictConfig, trial: optuna.Trial) -> None:
    base_run_name = OmegaConf.select(config, "mlflow.run_name")
    if base_run_name is None:
        base_run_name = tuning_cfg.study_name
    OmegaConf.update(
        config,
        "mlflow.run_name",
        f"{base_run_name}_trial_{trial.number:03d}",
        merge=False,
    )

    existing_tags = OmegaConf.select(config, "mlflow.tags")
    tags = {}
    if existing_tags is not None:
        tags = dict(OmegaConf.to_container(existing_tags, resolve=True))
    tags.update({
        "study_name": str(tuning_cfg.study_name),
        "trial_number": str(trial.number),
        "objective_metric": str(tuning_cfg.objective_metric),
    })
    OmegaConf.update(config, "mlflow.tags", tags, merge=True)


def _log_trial_configuration(
    config: DictConfig,
    *,
    display_trial_idx: int,
    trial_total: int,
    optuna_trial_number: int,
) -> None:
    tracked_paths = (
        "model.lambda_coef",
        "model.backbone.resnet_block.temperature",
        "model.backbone.resnet_block.init_mu",
        "model.backbone.resnet_block.sigma",
        "optimizer.lr",
        "optimizer.weight_decay",
        "dataloaders.batch_size",
    )
    parts: list[str] = []
    for path in tracked_paths:
        value = OmegaConf.select(config, path)
        if value is not None:
            parts.append(f"{path}={value}")
    if not parts:
        parts.append("no tracked params")
    tqdm.write(
        _format_tuning_progress(
            display_trial_idx=display_trial_idx,
            trial_total=trial_total,
            optuna_trial_number=optuna_trial_number,
        )
        + " | params | "
        + " | ".join(parts)
    )


def _update_repeat_metadata(
    config: DictConfig,
    *,
    repeat_number: int,
    repeat_total: int,
    repeat_seed: int | None,
    attempt_number: int,
) -> None:
    base_run_name = OmegaConf.select(config, "mlflow.run_name")
    if base_run_name is not None:
        OmegaConf.update(
            config,
            "mlflow.run_name",
            f"{base_run_name}_repeat_{repeat_number:02d}_attempt_{attempt_number:02d}",
            merge=False,
        )

    existing_tags = OmegaConf.select(config, "mlflow.tags")
    tags = {}
    if existing_tags is not None:
        tags = dict(OmegaConf.to_container(existing_tags, resolve=True))
    tags.update({
        "repeat_number": str(repeat_number),
        "repeat_total": str(repeat_total),
        "repeat_seed": str(repeat_seed) if repeat_seed is not None else "none",
        "attempt_number": str(attempt_number),
    })
    OmegaConf.update(config, "mlflow.tags", tags, merge=True)

    run_history_name = OmegaConf.select(config, "run_history.run_name")
    if run_history_name is not None:
        OmegaConf.update(
            config,
            "run_history.run_name",
            f"{run_history_name}_repeat_{repeat_number:02d}_attempt_{attempt_number:02d}",
            merge=False,
        )


def _build_restart_guard(
    tuning_cfg: DictConfig,
    *,
    objective_metric: str,
    direction: str,
) -> RepeatRestartGuard | None:
    cfg = getattr(tuning_cfg, "restart_guard", None)
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return None

    metric_name = str(getattr(cfg, "metric", objective_metric) or objective_metric)
    mode = str(getattr(cfg, "mode", "max" if direction == "maximize" else "min")).lower()
    epoch = int(getattr(cfg, "epoch"))
    threshold = float(getattr(cfg, "threshold"))
    return RepeatRestartGuard(
        metric_name=metric_name,
        mode=mode,
        epoch=epoch,
        threshold=threshold,
    )


def _compose_epoch_end_callbacks(*callbacks):
    def _callback(epoch, train_metrics, valid_metrics, model, optimizer, run_history):
        for callback in callbacks:
            if callback is None:
                continue
            callback(epoch, train_metrics, valid_metrics, model, optimizer, run_history)

    return _callback


def _resolve_output_dir(tuning_cfg: DictConfig) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    configured = Path(str(getattr(tuning_cfg, "output_dir", "outputs/studies")))
    return configured if configured.is_absolute() else repo_root / configured


def _create_unique_study_dir(tuning_cfg: DictConfig) -> Path:
    root_dir = _resolve_output_dir(tuning_cfg)
    root_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = str(getattr(tuning_cfg, "study_name", "optuna"))
    study_dir = root_dir / f"{timestamp}_{study_name}"
    suffix = 1
    while study_dir.exists():
        study_dir = root_dir / f"{timestamp}_{study_name}_{suffix}"
        suffix += 1
    study_dir.mkdir(parents=True, exist_ok=False)
    return study_dir


def _resolve_study_dir(tuning_cfg: DictConfig) -> Path:
    if HydraConfig.initialized():
        output_dir = getattr(HydraConfig.get().runtime, "output_dir", None)
        if output_dir is not None:
            study_dir = Path(str(output_dir))
            study_dir.mkdir(parents=True, exist_ok=True)
            return study_dir
    return _create_unique_study_dir(tuning_cfg)


def _study_runs_dir(study_dir: Path) -> Path:
    return study_dir / "runs"


def _set_trial_run_history_root(config: DictConfig, study_dir: Path) -> None:
    OmegaConf.update(
        config,
        "run_history.root_dir",
        str(_study_runs_dir(study_dir)),
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        config,
        "run_history.use_hydra_output_dir",
        False,
        merge=False,
        force_add=True,
    )


def _apply_cli_tuning_overrides(config: DictConfig) -> None:
    for path, value in CLI_TUNING_OVERRIDES.items():
        OmegaConf.update(config, path, value, merge=False)


def _resolve_tuning_mode(tuning_cfg: DictConfig) -> str:
    mode = str(getattr(tuning_cfg, "mode", "optuna")).lower()
    if mode not in {"optuna", "grid"}:
        raise ValueError("config.tuning.mode must be either 'optuna' or 'grid'.")
    return mode


class TrialObserver:
    def __init__(self, trial: optuna.Trial, metric_name: str, direction: str, *, allow_pruning: bool = True):
        self.trial = trial
        self.metric_name = metric_name
        self.direction = direction
        self.allow_pruning = allow_pruning
        self.best_value: float | None = None
        self.best_epoch: int | None = None

    def __call__(
        self,
        epoch: int,
        train_metrics,
        valid_metrics,
        model,
        optimizer,
        run_history,
    ) -> None:
        if self.metric_name not in valid_metrics:
            available_metrics = ", ".join(sorted(valid_metrics.keys()))
            raise KeyError(
                f"Objective metric '{self.metric_name}' is missing in validation metrics. "
                f"Available metrics: {available_metrics}"
            )

        value = float(valid_metrics[self.metric_name])
        if self.best_value is None:
            improved = True
        elif self.direction == "maximize":
            improved = value > self.best_value
        else:
            improved = value < self.best_value

        if improved:
            self.best_value = value
            self.best_epoch = epoch

        if not self.allow_pruning:
            return

        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} with {self.metric_name}={value:.6f}"
            )


def _summarize_study(
    study: optuna.Study,
    study_dir: Path,
    config: DictConfig,
    *,
    mode: str,
    effective_trials: int,
    grid_total_trials: int | None,
) -> None:
    resolved_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    study_config_path = study_dir / "study_config.yaml"
    OmegaConf.save(config=resolved_config, f=str(study_config_path))

    trials_df = study.trials_dataframe()
    trials_df.to_csv(study_dir / "trials.csv", index=False)

    best_trial = None
    best_value = None
    best_params: dict[str, Any] = {}
    try:
        best_trial = study.best_trial
        best_value = study.best_value
        best_params = dict(best_trial.user_attrs.get("suggested_params") or study.best_params)
    except ValueError:
        pass

    resolved_search_space = OmegaConf.select(config, "tuning.search_space")
    resolved_grid_points = OmegaConf.select(config, "tuning.points")
    best_trial_path = study_dir / "best_trial.yaml"
    if best_trial is not None:
        best_trial_payload: dict[str, Any] = {
            "trial_number": best_trial.number,
            "objective_value": best_value,
            "params": dict(best_params),
            "grid_point_index": best_trial.user_attrs.get("grid_point_index"),
            "best_epoch": best_trial.user_attrs.get("best_epoch"),
            "best_repeat_number": best_trial.user_attrs.get("best_repeat_number"),
            "best_repeat_seed": best_trial.user_attrs.get("best_repeat_seed"),
            "run_id": best_trial.user_attrs.get("run_id"),
            "run_dir": best_trial.user_attrs.get("run_dir"),
            "repeat_results": best_trial.user_attrs.get("repeat_results"),
            "repeat_failures": best_trial.user_attrs.get("repeat_failures"),
            "repeat_restarts": best_trial.user_attrs.get("repeat_restarts"),
        }
        run_dir = best_trial.user_attrs.get("run_dir")
        if run_dir is not None:
            run_config_path = Path(str(run_dir)) / "config_resolved.yaml"
            if run_config_path.exists():
                best_trial_payload["config_path"] = str(run_config_path)
                best_trial_payload["config"] = OmegaConf.to_container(
                    OmegaConf.load(run_config_path),
                    resolve=False,
                )
        OmegaConf.save(config=OmegaConf.create(best_trial_payload), f=str(best_trial_path))

    summary = {
        "study_name": study.study_name,
        "mode": mode,
        "direction": study.direction.name.lower(),
        "objective_metric": OmegaConf.select(config, "tuning.objective_metric"),
        "requested_trials": OmegaConf.select(config, "tuning.n_trials") if mode == "optuna" else None,
        "effective_trials": effective_trials,
        "grid_total_trials": grid_total_trials,
        "repeats_per_trial": OmegaConf.select(config, "tuning.repeats_per_trial"),
        "seed_base": OmegaConf.select(config, "tuning.seed_base"),
        "seed_stride": OmegaConf.select(config, "tuning.seed_stride"),
        "restart_guard": OmegaConf.to_container(
            OmegaConf.select(config, "tuning.restart_guard"),
            resolve=True,
        ) if OmegaConf.select(config, "tuning.restart_guard") is not None else None,
        "best_trial_number": best_trial.number if best_trial is not None else None,
        "best_value": best_value,
        "best_params": best_params,
        "best_epoch": best_trial.user_attrs.get("best_epoch") if best_trial is not None else None,
        "best_repeat_number": best_trial.user_attrs.get("best_repeat_number") if best_trial is not None else None,
        "best_repeat_seed": best_trial.user_attrs.get("best_repeat_seed") if best_trial is not None else None,
        "best_run_id": best_trial.user_attrs.get("run_id") if best_trial is not None else None,
        "best_run_dir": best_trial.user_attrs.get("run_dir") if best_trial is not None else None,
        "search_space": (
            OmegaConf.to_container(resolved_search_space, resolve=True)
            if resolved_search_space is not None
            else {}
        ),
        "grid_points": (
            OmegaConf.to_container(resolved_grid_points, resolve=True)
            if resolved_grid_points is not None
            else []
        ),
        "completed_trials": sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE),
        "pruned_trials": sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED),
        "failed_trials": sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL),
        "study_dir": str(study_dir),
        "runs_dir": str(_study_runs_dir(study_dir)),
        "study_config_path": str(study_config_path),
        "best_trial_path": str(best_trial_path) if best_trial is not None else None,
    }
    (study_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_study(
    config: DictConfig,
    *,
    mode: str,
    grid_search_space: dict[str, list[Any]] | None = None,
) -> optuna.Study:
    tuning_cfg = config.tuning
    if mode == "grid":
        if not grid_search_space:
            raise ValueError("Grid search mode requires a non-empty search space or explicit points.")
        sampler = optuna.samplers.GridSampler(grid_search_space)
        pruner = (
            instantiate(tuning_cfg.pruner)
            if getattr(tuning_cfg, "pruner", None)
            else optuna.pruners.NopPruner()
        )
    else:
        sampler = instantiate(tuning_cfg.sampler) if getattr(tuning_cfg, "sampler", None) else None
        pruner = instantiate(tuning_cfg.pruner) if getattr(tuning_cfg, "pruner", None) else None
    return optuna.create_study(
        study_name=str(tuning_cfg.study_name),
        direction=str(tuning_cfg.direction),
        sampler=sampler,
        pruner=pruner,
        storage=getattr(tuning_cfg, "storage", None),
        load_if_exists=bool(getattr(tuning_cfg, "load_if_exists", True)),
    )


@hydra.main(config_path=CONFIGS_PATH, config_name="tune", version_base=None)
def main(config: DictConfig) -> None:
    _apply_cli_tuning_overrides(config)
    tuning_cfg = config.tuning
    if not getattr(tuning_cfg, "enabled", False):
        raise ValueError("Tuning config is disabled. Use tuning=optuna or enable config.tuning.enabled.")

    mode = _resolve_tuning_mode(tuning_cfg)
    direction = str(tuning_cfg.direction).lower()
    objective_metric = str(tuning_cfg.objective_metric)
    if direction not in {"minimize", "maximize"}:
        raise ValueError("config.tuning.direction must be either 'minimize' or 'maximize'.")

    search_space_cfg = getattr(tuning_cfg, "search_space", None)
    grid_points = build_grid_points(getattr(tuning_cfg, "points", None)) if mode == "grid" else []
    has_search_space = bool(search_space_cfg)
    has_grid_points = bool(grid_points)

    if mode == "optuna":
        if not has_search_space:
            raise ValueError("config.tuning.search_space must be defined for optuna tuning runs.")
        if has_grid_points:
            raise ValueError("config.tuning.points is only supported in grid mode.")
    else:
        if has_search_space == has_grid_points:
            raise ValueError(
                "Grid tuning requires exactly one of config.tuning.search_space or config.tuning.points."
            )

    grid_search_space = None
    if mode == "grid":
        if has_grid_points:
            grid_search_space = build_point_grid_search_space(grid_points)
        elif has_search_space:
            grid_search_space = build_grid_search_space(search_space_cfg)
    effective_trials = (
        len(grid_points)
        if has_grid_points
        else count_grid_trials(grid_search_space)
        if grid_search_space is not None
        else int(getattr(tuning_cfg, "n_trials", 1))
    )
    repeats_per_trial = int(getattr(tuning_cfg, "repeats_per_trial", 1))
    repeat_seeds = resolve_repeat_seeds(
        repeats_per_trial,
        seed_base=getattr(tuning_cfg, "seed_base", None),
        seed_stride=int(getattr(tuning_cfg, "seed_stride", 1)),
    )
    repeat_seed_stride = int(getattr(tuning_cfg, "seed_stride", 1))
    restart_max_attempts = int(getattr(tuning_cfg, "restart_guard", {}).get("max_attempts_per_repeat", 1))
    if restart_max_attempts <= 0:
        raise ValueError("tuning.restart_guard.max_attempts_per_repeat must be >= 1.")
    restart_guard_template = _build_restart_guard(
        tuning_cfg,
        objective_metric=objective_metric,
        direction=direction,
    )
    restart_guard_enabled = restart_guard_template is not None

    study_dir = _resolve_study_dir(tuning_cfg)
    _study_runs_dir(study_dir).mkdir(parents=True, exist_ok=True)
    if mode == "grid":
        print(
            f"Grid search artifacts and trial runs: {study_dir} | points={effective_trials} | "
            f"repeats={repeats_per_trial}"
        )
    else:
        print(f"Optuna artifacts and trial runs: {study_dir} | repeats={repeats_per_trial}")
    study = _build_study(config, mode=mode, grid_search_space=grid_search_space)

    def objective(trial: optuna.Trial) -> float:
        display_trial_idx = trial.number + 1
        trial_total = effective_trials
        trial_config = _clone_config(config)
        if mode == "grid":
            if has_grid_points:
                point_index, suggested_params = _apply_grid_points(trial, trial_config, grid_points)
                trial.set_user_attr("grid_point_index", point_index)
            else:
                suggested_params = _apply_grid_search_space(trial, trial_config, grid_search_space or {})
        else:
            suggested_params = _apply_optuna_search_space(trial, trial_config, tuning_cfg.search_space)
        if LAMBDA_CONFIG_PATH in suggested_params:
            _assert_lambda_matches_config(
                trial_config,
                suggested_params[LAMBDA_CONFIG_PATH],
                source=f"grid_params['{LAMBDA_CONFIG_PATH}']",
            )
        if LAMBDA_CONFIG_PATH in trial.params:
            _assert_lambda_matches_config(
                trial_config,
                trial.params[LAMBDA_CONFIG_PATH],
                source=f"trial.params['{LAMBDA_CONFIG_PATH}']",
            )
        _update_trial_metadata(trial_config, tuning_cfg, trial)
        _set_trial_run_history_root(trial_config, study_dir)
        _log_trial_configuration(
            trial_config,
            display_trial_idx=display_trial_idx,
            trial_total=trial_total,
            optuna_trial_number=trial.number,
        )

        trial.set_user_attr("suggested_params", suggested_params)
        trial.set_user_attr("display_trial_idx", display_trial_idx)
        repeat_results: list[dict[str, Any]] = []
        repeat_failures: list[dict[str, Any]] = []
        repeat_restarts: list[dict[str, Any]] = []

        for repeat_index, repeat_seed in enumerate(repeat_seeds, start=1):
            for attempt_index in range(1, restart_max_attempts + 1):
                attempt_seed = resolve_repeat_attempt_seed(
                    repeat_seed,
                    attempt_number=attempt_index,
                    repeats_per_trial=repeats_per_trial,
                    seed_stride=repeat_seed_stride,
                )
                repeat_config = _clone_config(trial_config)
                OmegaConf.update(repeat_config, "seed", attempt_seed, merge=False)
                _update_repeat_metadata(
                    repeat_config,
                    repeat_number=repeat_index,
                    repeat_total=repeats_per_trial,
                    repeat_seed=attempt_seed,
                    attempt_number=attempt_index,
                )
                repeat_observer = TrialObserver(
                    trial,
                    objective_metric,
                    direction,
                    allow_pruning=(
                        mode == "optuna"
                        and repeats_per_trial == 1
                        and not restart_guard_enabled
                    ),
                )
                repeat_guard = _build_restart_guard(
                    tuning_cfg,
                    objective_metric=objective_metric,
                    direction=direction,
                ) if restart_guard_enabled else None
                progress_context = {
                    "trial_number": display_trial_idx,
                    "display_trial_idx": display_trial_idx,
                    "trial_total": trial_total,
                    "optuna_trial_number": trial.number,
                    "repeat_number": repeat_index,
                    "repeat_total": repeats_per_trial,
                    "attempt_number": attempt_index,
                    "attempt_total": restart_max_attempts,
                    "grid_params": dict(suggested_params),
                    "optuna_trial_params": dict(trial.params),
                }
                epoch_callback = _compose_epoch_end_callbacks(repeat_observer, repeat_guard)

                try:
                    result = run_training(
                        repeat_config,
                        epoch_end_callback=epoch_callback,
                        progress_context=progress_context,
                    )
                except RepeatRestartRequested as exc:
                    restart_event = {
                        "repeat_number": repeat_index,
                        "attempt_number": attempt_index,
                        "seed": attempt_seed,
                        "metric_name": exc.metric_name,
                        "epoch": exc.epoch,
                        "value": exc.value,
                        "threshold": exc.threshold,
                        "run_dir": exc.run_dir,
                        "run_id": exc.run_id,
                    }
                    repeat_restarts.append(restart_event)
                    tqdm.write(
                        _format_tuning_progress(
                            display_trial_idx=display_trial_idx,
                            trial_total=trial_total,
                            optuna_trial_number=trial.number,
                            repeat_number=repeat_index,
                            repeat_total=repeats_per_trial,
                            attempt_number=attempt_index,
                            attempt_total=restart_max_attempts,
                        )
                        + " | restart | "
                        f"{exc.metric_name}={exc.value:.6f} at epoch {exc.epoch} | "
                        f"threshold={exc.threshold:.6f} | seed={attempt_seed}"
                    )
                    if attempt_index == restart_max_attempts:
                        repeat_failures.append({
                            "repeat_number": repeat_index,
                            "seed": attempt_seed,
                            "attempt_number": attempt_index,
                            "error": str(exc),
                            "reason": "restart_guard",
                        })
                    continue
                except optuna.TrialPruned:
                    if repeat_observer.best_value is not None:
                        trial.set_user_attr("best_epoch", repeat_observer.best_epoch)
                        trial.set_user_attr("best_objective_value", repeat_observer.best_value)
                        tqdm.write(
                            _format_tuning_progress(
                                display_trial_idx=display_trial_idx,
                                trial_total=trial_total,
                                optuna_trial_number=trial.number,
                                repeat_number=repeat_index,
                                repeat_total=repeats_per_trial,
                            )
                            + f" | pruned | best {objective_metric}={repeat_observer.best_value:.6f} | "
                            f"epoch={repeat_observer.best_epoch}"
                        )
                    else:
                        tqdm.write(
                            _format_tuning_progress(
                                display_trial_idx=display_trial_idx,
                                trial_total=trial_total,
                                optuna_trial_number=trial.number,
                                repeat_number=repeat_index,
                                repeat_total=repeats_per_trial,
                            )
                            + " | pruned before first valid metric"
                        )
                    raise
                except Exception as exc:
                    repeat_failures.append({
                        "repeat_number": repeat_index,
                        "seed": attempt_seed,
                        "attempt_number": attempt_index,
                        "error": str(exc),
                        "reason": "exception",
                    })
                    tqdm.write(
                        _format_tuning_progress(
                            display_trial_idx=display_trial_idx,
                            trial_total=trial_total,
                            optuna_trial_number=trial.number,
                            repeat_number=repeat_index,
                            repeat_total=repeats_per_trial,
                        )
                        + f" | failed | seed={attempt_seed} | error={exc}"
                    )
                    break
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                objective_value = repeat_observer.best_value
                if objective_value is None:
                    last_valid_metrics = result.get("last_valid_metrics", {})
                    if objective_metric not in last_valid_metrics:
                        raise KeyError(
                            f"Objective metric '{objective_metric}' is missing in final validation metrics."
                        )
                    objective_value = float(last_valid_metrics[objective_metric])

                repeat_result = {
                    "repeat_number": repeat_index,
                    "attempt_number": attempt_index,
                    "seed": attempt_seed,
                    "objective_value": float(objective_value),
                    "best_epoch": repeat_observer.best_epoch,
                    "run_id": result.get("run_id"),
                    "run_dir": result.get("run_dir"),
                    "best_metric_name": result.get("best_metric_name"),
                    "best_metric_value": result.get("best_metric_value"),
                }
                repeat_results.append(repeat_result)
                tqdm.write(
                    _format_tuning_progress(
                        display_trial_idx=display_trial_idx,
                        trial_total=trial_total,
                        optuna_trial_number=trial.number,
                        repeat_number=repeat_index,
                        repeat_total=repeats_per_trial,
                    )
                    + f" | completed | best {objective_metric}={objective_value:.6f} | "
                    f"epoch={repeat_observer.best_epoch} | "
                    f"seed={attempt_seed} | run_dir={result.get('run_dir')}"
                )
                break

        if not repeat_results:
            error_summary = "; ".join(
                f"repeat={failure['repeat_number']} attempt={failure.get('attempt_number')} "
                f"seed={failure['seed']} error={failure['error']}"
                for failure in repeat_failures
            )
            raise RuntimeError(f"All repeats failed for display trial {display_trial_idx}: {error_summary}")

        best_repeat = select_best_repeat(direction, repeat_results)
        trial.set_user_attr("run_id", best_repeat.get("run_id"))
        trial.set_user_attr("run_dir", best_repeat.get("run_dir"))
        trial.set_user_attr("best_epoch", best_repeat.get("best_epoch"))
        trial.set_user_attr("best_objective_value", best_repeat.get("objective_value"))
        trial.set_user_attr("best_repeat_number", best_repeat.get("repeat_number"))
        trial.set_user_attr("best_repeat_seed", best_repeat.get("seed"))
        trial.set_user_attr("repeat_results", repeat_results)
        trial.set_user_attr("repeat_failures", repeat_failures)
        trial.set_user_attr("repeat_restarts", repeat_restarts)
        trial.set_user_attr("repeat_seeds", [result["seed"] for result in repeat_results])
        trial.set_user_attr(
            "repeat_objective_values",
            [result["objective_value"] for result in repeat_results],
        )
        trial.set_user_attr("repeat_run_dirs", [result["run_dir"] for result in repeat_results])
        tqdm.write(
            _format_tuning_progress(
                display_trial_idx=display_trial_idx,
                trial_total=trial_total,
                optuna_trial_number=trial.number,
            )
            + f" | selected repeat {best_repeat['repeat_number']}/{repeats_per_trial} | "
            f"best {objective_metric}={best_repeat['objective_value']:.6f} | "
            f"seed={best_repeat['seed']} | run_dir={best_repeat.get('run_dir')}"
        )
        return float(best_repeat["objective_value"])

    try:
        study.optimize(
            objective,
            n_trials=effective_trials,
            timeout=getattr(tuning_cfg, "timeout", None),
            n_jobs=int(getattr(tuning_cfg, "n_jobs", 1)),
            gc_after_trial=True,
        )
    finally:
        _summarize_study(
            study,
            study_dir,
            config,
            mode=mode,
            effective_trials=effective_trials,
            grid_total_trials=(effective_trials if mode == "grid" else None),
        )


if __name__ == "__main__":
    initialize_cli_flags()
    main()
