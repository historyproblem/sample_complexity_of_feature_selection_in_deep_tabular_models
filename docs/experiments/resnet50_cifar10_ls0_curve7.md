# ResNet-50 / CIFAR-10: seven LS=0 threshold profiles

The manifest is `configs/experiment/pruning_v3/resnet50_cifar10_ls0_curve7.yaml`.
Each profile independently trains 60 adaptive-lambda search epochs and 90
physical ungated recovery epochs. Only `soft_drop` and `hard_drop` differ across
profiles, with `soft_drop = hard_drop / 2`. No new dense reference is trained.

| Planning size | soft_drop | hard_drop |
| ---: | ---: | ---: |
| 23M | 0.00005 | 0.0001 |
| 18M | 0.00025 | 0.0005 |
| 12M | 0.0005 | 0.001 |
| 8M | 0.0025 | 0.005 |
| 5M | 0.0075 | 0.015 |
| 4M | 0.015 | 0.03 |
| 2.9M floor probe | 0.05 | 0.10 |

These sizes are planning labels, not predicted measurements or pruning quotas.
The exact architecture floor is 2,876,538 parameters: pruning only bottleneck
internal channels cannot reach the requested 1.5M endpoint. The shared
`min_keep_ratio` is lowered from the historical anchor's 0.08 to 0.001 for all
seven profiles to explore the low end; this is the only shared recipe change.
The historical 12.728M measurement is not promised to repeat under this floor.

The recipe uses the training implementation from commit `10157c8`, with no
changes to training or model code: legacy linear open-probability/L1 penalty,
`initial_channels` normalization, `learned_closed_gates`, gate threshold 0.5,
entropy coefficient 0, plain CE (`label_smoothing=0`), seed 42, AdamW 0.001 /
weight decay 0.0005, and independent cosine schedules ending at zero. Recovery
starts from selected surviving weights with fresh optimizer/scheduler and
train-only BN calibration on 200 batches. Physical recovery has no gates; the
adaptive controller state is retained rather than used for a gate penalty.

## Run from the server repository root

The default immutable reference is the existing
`outputs/baseline/one_shot_dense_reference_seed42`. Its original zero-epoch
`shared_random_seed42.pt` and `J1_dense_control/{pilot_state.json,
global_history.csv,resolved_config.yaml}` must exist. Trained dense checkpoints
are unnecessary. Use `--reference-output PATH` for a different compatible copy.

```bash
git fetch origin
git switch feature/resnet50-cifar10-ls0-curve7
git pull --ff-only origin feature/resnet50-cifar10-ls0-curve7

.venv/bin/python scripts/launch_resnet50_cifar10_ls0_curve7.py --preflight-only
nohup .venv/bin/python -u scripts/launch_resnet50_cifar10_ls0_curve7.py > resnet50_ls0_curve7.log 2>&1 &
```

`--validate-only` checks all configurations and the structural floor on CPU
without accessing data. Full preflight checks the repository environment,
CUDA, reference compatibility and free space (55 GiB by default). Nested epoch
checkpoints are removed after their completed stage is verified; selected root
checkpoints and deployment artifacts are retained. `--keep-nested-checkpoints`
disables cleanup and requires substantially more disk space.

All seven validation rows, including failures, are written and locked before
test inference. Frozen official-test accuracy uses all 10,000 CIFAR-10 examples
without training or BN updates. Comparisons remain exploratory because earlier
test results informed this study.

Results are under `outputs/runs/resnet50_cifar10_ls0_curve7_v1/`:
`validation_results.csv`, `final_results.csv`, `study_state.json`, and
`study_summary.json`. A failed profile does not discard successful profiles'
test results. The process exits nonzero after writing results if any profile or
test fails. `--resume-completed` verifies/reuses completed stages and tests and
may run an unstarted recovery after a completed search. It never silently
restarts a partially trained stage or overwrites an incompatible study.
