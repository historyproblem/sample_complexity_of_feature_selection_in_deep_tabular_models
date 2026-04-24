# Sample Complexity of Feature Selection in Deep Tabular Models

| Title      | Sample Complexity of Feature Selection in Deep Tabular Models |
|------------|-----------------------------------------------------------------|
| Authors    | Khlystov Gregory, Vladislav Meshkov, Altay Eynullayev, Denis Rubtsov           |
| Advisor    | Andrey Grabovoy, PhD                                            |
| Consultant |                                                                 |

---

## 🔎 Idea and Novelty

In learning theory, sample complexity determines how much data is required for generalization. For linear models, classical results establish relationships between sample size, total dimensionality, and number of relevant features. However, for neural networks, the interplay between feature selection, data requirements, and model complexity remains poorly understood. Key open questions include: How does the required sample size scale with the number of irrelevant features when training neural networks? How do different feature selection methods affect this scaling? How do feature correlations impact the data efficiency of selection algorithms? This work aims to empirically investigate the sample complexity of modern feature selection methods for deep tabular models through controlled experiments.

---

## Abstract

*To be added.*

---

## Baselines

*To be added.*

---

## Useful links

*To be added.*

---

## Datasets

*To be added.*

### Installation

1. Clone the repository:
```bash
    git clone (TODO)
    cd sample_complexity_of_feature_selection_in_deep_tabular_models
    pip install -e .
```

2. Run a training example:
```bash
    python3 src/net_complexity/train.py
```

Pick a different recipe through Hydra defaults instead of maintaining a separate long CLI:
```bash
    python3 src/net_complexity/train.py experiment=stg_cifar10_120 seed=3
```

3. Run Optuna hyperparameter search:
```bash
    python3 src/net_complexity/tune.py
```

The tuning entrypoint uses the same experiment recipes:
```bash
    python3 src/net_complexity/tune.py experiment=stg_cifar10_120 tuning=stg_cifar10_optuna120
```

Tune search settings through Hydra overrides instead of a separate tuning CLI:

```bash
    python3 src/net_complexity/tune.py \
      experiment=stg_cifar10_120 \
      tuning=stg_cifar10_optuna120 \
      tuning.study_name=stg_quick_check \
      tuning.n_trials=20 \
      tuning.direction=maximize
```

Grid search is configured the same way:

```bash
    python3 src/net_complexity/tune.py \
      experiment=stg_cifar10_120 \
      tuning=stg_lambda_initmu_grid_sigma05 \
      tuning.repeats_per_trial=3 \
      --seed-base 42 \
      --seed-stride 1 \
      --restart-below-acc 20:0.4 \
      --restart-max-attempts 5
```

The remaining custom shortcuts are only for repeat-seed and restart-guard control:
- `--restart-below-acc EPOCH:THRESHOLD`
- `--restart-max-attempts N`
- `--seed-base N`
- `--seed-stride N`

Notes for grid mode:
- default mode is `optuna`
- grid mode expands the full discrete search space
- `tuning.repeats_per_trial=N` reruns each parameter point with different seeds and keeps the best repeat as the trial objective
- `--restart-below-acc 20:0.4` restarts a repeat with a new seed if `valid_accuracy` has not reached `0.4` by epoch `20`
- search spaces live in YAML under `configs/tuning`

Current configs are organized as:
- `configs/data`, `configs/model`, `configs/method`, `configs/train`, `configs/optimizer`, `configs/scheduler`, `configs/tracking`, `configs/run_history`, `configs/metrics`: reusable config layers
- `configs/experiment`: thin recipes that compose the layers above and keep only a few recipe-specific overrides
- `configs/tuning`: Optuna/grid settings and search spaces
- top-level `configs/train.yaml` and `configs/tune.yaml`: entry configs that default to one experiment recipe and one tuning profile
- `configs/old`: legacy configs kept for backward compatibility

Artifacts are split by intent:
- single training run: `outputs/runs/<timestamp>_<run_name>/` with scalar `history.csv`, per-channel `channel_history.csv.gz`, `config_resolved.yaml`, `summary.json`, `checkpoints/`, and `.hydra/`
- tuning study: `outputs/studies/<timestamp>_<study_name>/` with `study_config.yaml`, `trials.csv`, `summary.json`, `best_trial.yaml`, `runs/`, and `.hydra/`

Hydra metadata now lives inside the same run or study directory instead of a separate `outputs/YYYY-MM-DD/...` tree.

## 📊 Experiment Tracking with MLflow

To visualize and compare experiment results, run:

```bash
mlflow ui --port 5000
```
