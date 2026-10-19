# ResNet-50 / CIFAR-10: eight quarter-point validation bands

This study varies only the adaptive-lambda accuracy tolerances. All eight
independent models use the calibrated keep001 recipe: label smoothing 0,
internal-width L1 gates, minimum surviving fraction 0.001, 60 search epochs and
90 ungated recovery epochs. Search candidates are restricted to epochs 31-60;
each physical candidate receives 200 train-only BatchNorm calibration batches
before its validation score is used for selection. Recovery keeps fresh AdamW
and cosine scheduling. The automatic adaptive-lambda log step resolves to
0.9210340371976183 for the common 60-epoch search.

The dense validation history is pinned by SHA256 and must end at 94.14% on
epoch 150. No new dense reference is trained. Each model starts from the same
existing zero-epoch seed42 initializer.

## Accuracy bands and launch order

The requested final endpoint 93.91-94.14 spans 0.23 percentage points. This
manifest uses 93.89-94.14 so every interval is exactly 0.25 percentage points.
The strictest profile runs first. Names describe accuracy, not model size.

| Order | Profile | Final validation equivalent, % | soft_drop | hard_drop |
| --- | --- | --- | --- | --- |
| 1 | val_9389_9414 | 93.89-94.14 | 0 | 0.0025 |
| 2 | val_9364_9389 | 93.64-93.89 | 0.0025 | 0.005 |
| 3 | val_9339_9364 | 93.39-93.64 | 0.005 | 0.0075 |
| 4 | val_9314_9339 | 93.14-93.39 | 0.0075 | 0.01 |
| 5 | val_9289_9314 | 92.89-93.14 | 0.01 | 0.0125 |
| 6 | val_9264_9289 | 92.64-92.89 | 0.0125 | 0.015 |
| 7 | val_9239_9264 | 92.39-92.64 | 0.015 | 0.0175 |
| 8 | val_9214_9239 | 92.14-92.39 | 0.0175 | 0.02 |

These bands express tolerances relative to the final dense validation reference:
the lower edge is `reference - hard_drop`, the upper edge is
`reference - soft_drop`. The upper edge is a controller threshold, not a ceiling
on the recovered model's accuracy. The lower edge is the final feasibility
threshold; infeasible models remain explicitly marked in the results.

During search the adaptive controller compares raw gated validation with the
corresponding epoch of the dense history. For example, dense epoch60 is 92.32%,
so the strictest search-end band is 92.07-92.32%, not 93.89-94.14%.
Physical checkpoint selection uses post-BN validation separately. Model size is
measured after export; this grid does not enforce a parameter budget.

## Run from the repository root

Validate all phase configurations on CPU without training or loading datasets:

```bash
.venv/bin/python scripts/launch_resnet50_cifar10_ls0_curve7.py \
  --config configs/experiment/pruning_v3/resnet50_cifar10_ls0_keep001_quarterpp_grid8.yaml \
  --validate-only
```

Launch the complete sequential study, including official frozen test after all
validation results have been locked:

```bash
nohup .venv/bin/python -u scripts/launch_resnet50_cifar10_ls0_curve7.py \
  --config configs/experiment/pruning_v3/resnet50_cifar10_ls0_keep001_quarterpp_grid8.yaml \
  > resnet50_ls0_quarterpp_grid8.log 2>&1 &
```

Watch progress:

```bash
tail -f resnet50_ls0_quarterpp_grid8.log
```

Results are written to
`outputs/runs/resnet50_cifar10_ls0_keep001_quarterpp_grid8_v1/final_results.csv`.
Accuracy columns contain fractions. The CSV includes
`required_final_validation_accuracy`, `soft_final_validation_accuracy`,
`validation_accuracy`, `quality_feasible`, actual `physical_parameters`, and
the separate official `test_accuracy`. Test inference does not train or update
BatchNorm. Comparisons remain exploratory because earlier test results informed
the experiment series.

## Compute budget

Each model consumes 150 training epochs; eight models consume 1,200 epochs.
At one minute per training epoch this is approximately 20 hours, plus selection,
calibration, I/O and test. Candidate calibration adds 48,000 train-forward batches
in total, with no optimizer updates. No nine-hour runtime guarantee applies.
