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

Current configs are organized as:
- `configs/experiment`: compact runnable experiment configs
- `configs/tuning`: Optuna settings and search spaces
- `configs/old`: legacy configs kept for backward compatibility

## 📊 Experiment Tracking with MLflow

To visualize and compare experiment results, run:

```bash
mlflow ui --port 5000
```
