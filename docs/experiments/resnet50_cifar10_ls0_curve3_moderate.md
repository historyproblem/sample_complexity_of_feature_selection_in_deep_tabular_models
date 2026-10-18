# ResNet-50 / CIFAR-10: three moderate soft-hard profiles

Three new models cover the compact part of the curve. The completed v2
`large_18m` result already provides a 14,848,414-parameter comparison with
validation 94.20% and official exploratory test 93.72%; it requires no new
training or test inference. That result remains attributed to its original
v2 run and thresholds, rather than being inserted as a new completed profile.

| Planning size | soft_drop | hard_drop | Hold-band width, percentage points |
| ---: | ---: | ---: | ---: |
| 3.7M floor probe | 0.05 | 0.06 | 1 |
| 6M | 0.015 | 0.018 | 0.3 |
| 10M | 0.0075 | 0.009 | 0.15 |

Compared with the corresponding v2 profiles, soft thresholds stay fixed and
the hold bands become five times narrower. The new queue has no profile with
the proposed 0.01-percentage-point band. It consumes 450 training epochs
across three independent models, each with the same 150-epoch budget.

The v2 recipe remains: adaptive lambda with automatic log_step
0.9210340371976183, gap_window 3, label smoothing 0, min_keep_ratio 0.08,
60 search + 90 recovery epochs, checkpoint selection from search epochs31-60,
seed42, fresh AdamW/cosine recovery and 200 training batches of BN calibration.
The immutable reference and zero-epoch initializer are reused. Gate and export
implementations are unchanged; this is a threshold-band study.

Planning sizes are not quotas or guarantees. The architecture minimum is
3,683,893 parameters, and actual deployment sizes are determined by search.
All three validation rows are locked before frozen official-test inference.
Comparisons with earlier test-informed experiments remain exploratory.

## Run from the server repository root

```bash
git fetch origin
git switch feature/resnet50-cifar10-ls0-curve7
git pull --ff-only origin feature/resnet50-cifar10-ls0-curve7

nohup .venv/bin/python -u scripts/launch_resnet50_cifar10_ls0_curve3_moderate.py \
  > resnet50_ls0_curve3_moderate_v1.log 2>&1 &
```

Use this command instead of the four-profile narrow command. An existing
server process is not stopped by updating the repository or starting a new
launcher; inspect the current queue before starting another GPU process.

```bash
tail -n 100 -f resnet50_ls0_curve3_moderate_v1.log
```

The manifest is `configs/experiment/pruning_v3/resnet50_cifar10_ls0_curve3_moderate.yaml`;
new artifacts go to `outputs/runs/resnet50_cifar10_ls0_curve3_moderate_v1/`.
`--validate-only` performs the CPU recipe/floor checks without loading data.
Actual launch verifies reference/environment inputs. Existing checkpoint
cleanup, failure recording and strict `--resume-completed` checks are preserved.
The final CSV contains only the three new runs; the completed 14.85M point
remains in `outputs/runs/resnet50_cifar10_ls0_curve7_v2/final_results.csv`.
