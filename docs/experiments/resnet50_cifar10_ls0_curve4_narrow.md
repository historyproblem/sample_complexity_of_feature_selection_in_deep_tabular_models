# ResNet-50 / CIFAR-10: four narrow soft-hard profiles

This study keeps the v2 search/recovery recipe and lowers each hard tolerance
towards its existing soft tolerance. The soft/hard values are independent
manifest fields. The lambda step remains automatic at 0.9210340371976183 for
the same 60-epoch search; this study changes only the tolerance bands.

The four pairs run sequentially, smallest planning size first. Each model uses
60 adaptive-lambda search epochs plus 90 physical recovery epochs, 150 total.

| Planning size | soft_drop | hard_drop | Hold-band width, percentage points |
| ---: | ---: | ---: | ---: |
| 3.7M floor probe | 0.05 | 0.055 | 0.5 |
| 6M | 0.015 | 0.0165 | 0.15 |
| 10M | 0.0075 | 0.00825 | 0.075 |
| 15M | 0.00025 | 0.00035 | 0.01 |

The last band stays wider than one count quantum of the three-epoch mean on
5,000 validation examples (about 0.00667 percentage points). The planning
sizes are labels, not quotas or checkpoint-selection inputs. Actual sizes and
their order are search outcomes; the run can produce a model larger than 15M.
With the unchanged 8% boundary floor, the architecture minimum is exactly
3,683,893 parameters. No profile claims a physically unreachable 1.5M model.

Common settings: label smoothing 0, min_keep_ratio 0.08, search selection only
from epochs 31-60 for feasible and fallback candidates, L1 open-probability
penalty, tau 0.5, entropy coefficient 0, seed 42, AdamW lr 0.001 / weight decay
0.0005, fresh optimizer and cosine scheduler for recovery, and train-only BN
calibration on 200 batches before recovery. Adaptive lambda stays enabled;
physical recovery holds its state and has no gate penalty.

Search/export behavior is inherited from v2, including raw gated quality
feedback and the floor-constrained physical mask. The existing diagnostics
record their difference when closed channels must be retained by the floor.
This threshold study does not modify gate or weight-transfer implementation.

The immutable dense-reference history and zero-epoch initializer come from
the same `outputs/baseline/one_shot_dense_reference_seed42` directory as v2;
there is no new dense training. See the [v2 recipe](resnet50_cifar10_ls0_curve7.md)
for reference inputs and cleanup/resume behavior.

## Run from the server repository root

```bash
git fetch origin
git switch feature/resnet50-cifar10-ls0-curve7
git pull --ff-only origin feature/resnet50-cifar10-ls0-curve7

.venv/bin/python scripts/launch_resnet50_cifar10_ls0_curve4_narrow.py --validate-only
nohup .venv/bin/python -u scripts/launch_resnet50_cifar10_ls0_curve4_narrow.py \
  > resnet50_ls0_curve4_narrow_v1.log 2>&1 &
```

`--validate-only` checks configuration and architecture on CPU without reading
data. Actual launch first checks the reference, CUDA and disk space. Read logs:

```bash
tail -n 100 -f resnet50_ls0_curve4_narrow_v1.log
```

The manifest is `configs/experiment/pruning_v3/resnet50_cifar10_ls0_curve4_narrow.yaml`.
Outputs go to `outputs/runs/resnet50_cifar10_ls0_curve4_narrow_v1/`, separately
from completed series. Four validation rows are locked before any frozen
official test evaluation; test uses all 10,000 examples with no training or BN
updates. Comparisons remain exploratory. Result files are `validation_results.csv`,
`final_results.csv`, `study_state.json` and `study_summary.json`.

`--resume-completed` reuses only verified completed stages/tests and refuses
changed thresholds, references or partial-stage retraining. A failed profile
or test is recorded while other profiles continue, then the process exits
nonzero after saving the available results.
