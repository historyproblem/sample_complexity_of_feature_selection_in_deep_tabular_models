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
from net_complexity.tuning.search import build_grid_search_space, count_grid_trials


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
        "best_trial_number": best_trial.number if best_trial is not None else None,
        "best_value": best_value,
        "best_params": best_params,
        "best_epoch": best_trial.user_attrs.get("best_epoch") if best_trial is not None else None,
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


@hydra.main(config_path="../../../configs/", config_name="tune", version_base=None)
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

    study_dir = _create_study_dir(tuning_cfg)
    if mode == "grid":
        print(f"Grid search artifacts: {study_dir} | points={effective_trials}")
    else:
        print(f"Optuna artifacts: {study_dir}")
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
        observer = TrialObserver(
            trial,
            objective_metric,
            direction,
            allow_pruning=(mode == "optuna"),
        )
        progress_context = {
            "trial_number": trial_number,
            "trial_total": trial_total,
        }

        trial.set_user_attr("suggested_params", suggested_params)
        try:
            result = run_training(
                trial_config,
                epoch_end_callback=observer,
                progress_context=progress_context,
            )
        except optuna.TrialPruned:
            if observer.best_value is not None:
                trial.set_user_attr("best_epoch", observer.best_epoch)
                trial.set_user_attr("best_objective_value", observer.best_value)
                tqdm.write(
                    f"[trial {trial_number}/{trial_total}] pruned | "
                    f"best {objective_metric}={observer.best_value:.6f} | "
                    f"epoch={observer.best_epoch}"
                )
            else:
                tqdm.write(f"[trial {trial_number}/{trial_total}] pruned before first valid metric")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        objective_value = observer.best_value
        if objective_value is None:
            last_valid_metrics = result.get("last_valid_metrics", {})
            if objective_metric not in last_valid_metrics:
                raise KeyError(
                    f"Objective metric '{objective_metric}' is missing in final validation metrics."
                )
            objective_value = float(last_valid_metrics[objective_metric])

        trial.set_user_attr("run_id", result.get("run_id"))
        trial.set_user_attr("run_dir", result.get("run_dir"))
        trial.set_user_attr("best_epoch", observer.best_epoch)
        trial.set_user_attr("best_objective_value", objective_value)
        tqdm.write(
            f"[trial {trial_number}/{trial_total}] completed | "
            f"best {objective_metric}={objective_value:.6f} | "
            f"epoch={observer.best_epoch} | "
            f"run_dir={result.get('run_dir')}"
        )
        return float(objective_value)

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
