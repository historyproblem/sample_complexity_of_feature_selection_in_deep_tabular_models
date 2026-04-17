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
from net_complexity.tuning.search import build_grid_search_space, count_grid_trials


CONFIGS_PATH = str(Path(__file__).resolve().parents[3] / "configs")


CLI_SEARCH_RESET = False
CLI_SEARCH_SPACE: dict[str, dict[str, Any]] = {}
CLI_TUNING_OVERRIDES: dict[str, Any] = {}


def initialize_cli_flags() -> None:
    global CLI_SEARCH_RESET, CLI_SEARCH_SPACE, CLI_TUNING_OVERRIDES
    CLI_SEARCH_RESET, CLI_SEARCH_SPACE, CLI_TUNING_OVERRIDES = install_tune_cli_flags()


def _clone_config(config: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


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
    configured = Path(str(getattr(tuning_cfg, "output_dir", "outputs/tuning")))
    return configured if configured.is_absolute() else repo_root / configured


def _create_study_dir(tuning_cfg: DictConfig) -> Path:
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


def _apply_cli_search_flags(config: DictConfig) -> None:
    if not CLI_SEARCH_RESET and not CLI_SEARCH_SPACE:
        return

    existing_search_space = {}
    if not CLI_SEARCH_RESET:
        current = OmegaConf.select(config, "tuning.search_space")
        if current is not None:
            existing_search_space = dict(OmegaConf.to_container(current, resolve=True))

    existing_search_space.update(CLI_SEARCH_SPACE)
    OmegaConf.update(config, "tuning.search_space", existing_search_space, merge=False)


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
    OmegaConf.save(config=resolved_config, f=str(study_dir / "config_resolved.yaml"))

    trials_df = study.trials_dataframe()
    trials_df.to_csv(study_dir / "trials.csv", index=False)

    best_trial = None
    best_value = None
    best_params: dict[str, Any] = {}
    try:
        best_trial = study.best_trial
        best_value = study.best_value
        best_params = study.best_params
    except ValueError:
        pass

    resolved_search_space = OmegaConf.select(config, "tuning.search_space")

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
        "completed_trials": sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE),
        "pruned_trials": sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED),
        "failed_trials": sum(1 for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL),
        "study_dir": str(study_dir),
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
            raise ValueError("Grid search mode requires a non-empty search space.")
        sampler = optuna.samplers.GridSampler(grid_search_space)
        pruner = None
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
    _apply_cli_search_flags(config)
    tuning_cfg = config.tuning
    if not getattr(tuning_cfg, "enabled", False):
        raise ValueError("Tuning config is disabled. Use tuning=optuna or enable config.tuning.enabled.")
    if not getattr(tuning_cfg, "search_space", None):
        raise ValueError("config.tuning.search_space must be defined for tuning runs.")

    mode = _resolve_tuning_mode(tuning_cfg)
    direction = str(tuning_cfg.direction).lower()
    objective_metric = str(tuning_cfg.objective_metric)
    if direction not in {"minimize", "maximize"}:
        raise ValueError("config.tuning.direction must be either 'minimize' or 'maximize'.")

    grid_search_space = build_grid_search_space(tuning_cfg.search_space) if mode == "grid" else None
    effective_trials = (
        count_grid_trials(grid_search_space)
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

    study_dir = _create_study_dir(tuning_cfg)
    if mode == "grid":
        print(
            f"Grid search artifacts: {study_dir} | points={effective_trials} | "
            f"repeats={repeats_per_trial}"
        )
    else:
        print(f"Optuna artifacts: {study_dir} | repeats={repeats_per_trial}")
    study = _build_study(config, mode=mode, grid_search_space=grid_search_space)

    def objective(trial: optuna.Trial) -> float:
        trial_number = trial.number + 1
        trial_total = effective_trials
        trial_config = _clone_config(config)
        if mode == "grid":
            suggested_params = _apply_grid_search_space(trial, trial_config, grid_search_space or {})
        else:
            suggested_params = _apply_optuna_search_space(trial, trial_config, tuning_cfg.search_space)
        _update_trial_metadata(trial_config, tuning_cfg, trial)

        trial.set_user_attr("suggested_params", suggested_params)
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
                    "trial_number": trial_number,
                    "trial_total": trial_total,
                    "repeat_number": repeat_index,
                    "repeat_total": repeats_per_trial,
                    "attempt_number": attempt_index,
                    "attempt_total": restart_max_attempts,
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
                        f"[trial {trial_number}/{trial_total} repeat {repeat_index}/{repeats_per_trial} "
                        f"attempt {attempt_index}/{restart_max_attempts}] restart | "
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
                            f"[trial {trial_number}/{trial_total} repeat {repeat_index}/{repeats_per_trial}] "
                            f"pruned | best {objective_metric}={repeat_observer.best_value:.6f} | "
                            f"epoch={repeat_observer.best_epoch}"
                        )
                    else:
                        tqdm.write(
                            f"[trial {trial_number}/{trial_total} repeat {repeat_index}/{repeats_per_trial}] "
                            "pruned before first valid metric"
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
                        f"[trial {trial_number}/{trial_total} repeat {repeat_index}/{repeats_per_trial}] "
                        f"failed | seed={attempt_seed} | error={exc}"
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
                    f"[trial {trial_number}/{trial_total} repeat {repeat_index}/{repeats_per_trial}] "
                    f"completed | best {objective_metric}={objective_value:.6f} | "
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
            raise RuntimeError(f"All repeats failed for trial {trial_number}: {error_summary}")

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
            f"[trial {trial_number}/{trial_total}] selected repeat {best_repeat['repeat_number']}/"
            f"{repeats_per_trial} | best {objective_metric}={best_repeat['objective_value']:.6f} | "
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
