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

3. Run Optuna hyperparameter search:
```bash
    python3 src/net_complexity/tune.py
```

You can override the most common tuning options from short CLI flags:

```bash
    python3 src/net_complexity/tune.py \
      --optuna \
      --study-name gumbel_quick_check \
      --trials 20 \
      --metric valid_accuracy \
      --maximize \
      --search-reset \
      --float "lambda=0.01:5.0:log" \
      --float "lr=0.0001:0.01:log" \
      --float "wd=0.000001:0.01:log" \
      --cat "bs=128,256,512"
```

Switch to exhaustive grid search with the same entrypoint:

```bash
    python3 src/net_complexity/tune.py \
      --grid \
      --study-name gumbel_grid \
      --repeats 3 \
      --seed-base 42 \
      --seed-stride 1 \
      --search-reset \
      --float "lambda=0.01:0.05:step=0.01" \
      --float "lr=0.0005:0.0015:step=0.0005" \
      --cat "bs=128,256"
```

Short search aliases:
- `lambda` -> `model.lambda_coef`
- `lr` -> `optimizer.lr`
- `wd` -> `optimizer.weight_decay`
- `bs` -> `dataloaders.batch_size`

Supported search flag formats:
- `--float "name=low:high[:log][:step=value]"`
- `--int "name=low:high[:log][:step=value]"`
- `--cat "name=value1,value2,..."`
- `--search "path=float:low:high[:log][:step=value]"` for explicit full config paths
- `--search-reset` clears the default YAML `tuning.search_space` before applying CLI flags

Useful tuning override flags:
- `--grid` / `--optuna`
- `--mode NAME`
- `--trials N`
- `--repeats N`
- `--restart-below-acc EPOCH:THRESHOLD`
- `--restart-max-attempts N`
- `--metric NAME`
- `--study-name NAME`
- `--jobs N`
- `--timeout SECONDS`
- `--output-dir PATH`
- `--seed-base N`
- `--seed-stride N`
- `--maximize` / `--minimize`

Notes for grid mode:
- default mode is `optuna`
- grid mode expands the full discrete search space
- `--repeats N` reruns each parameter point with different seeds and keeps the best repeat as the trial objective
- `--restart-below-acc 20:0.4` restarts a repeat with a new seed if `valid_accuracy` has not reached `0.4` by epoch `20`
- `float` grid ranges require `step=...`
- log-scaled numeric ranges are supported in `optuna` mode, but not in `grid`
- for arbitrary numeric grids, prefer `--cat "name=v1,v2,v3"`

Current configs are organized as:
- `configs/experiment`: compact runnable experiment configs
- `configs/tuning`: Optuna settings and search spaces
- `configs/old`: legacy configs kept for backward compatibility

## 📊 Experiment Tracking with MLflow

To visualize and compare experiment results, run:

```bash
mlflow ui --port 5000
```
