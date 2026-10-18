# ResNet-50 / CIFAR-10: nine-hour overnight queue

The queue contains three complete independent models, each using 60 search
and 90 recovery epochs. At one minute per epoch this is 450 minutes of
training, with another 90 minutes reserved for checkpoint selection, BN
calibration, I/O and frozen test. Four complete models would require ten hours
of training alone.

The previous seven-profile v2 archive recorded 1,057.63 minutes inside epoch
loops and 103.08 minutes outside them, about 14.73 extra minutes per model.
This supports planning three models for roughly 8-9 hours at the stated epoch
speed. Duration is an estimate: the launcher reports the budget but does not
kill an in-progress model at an exact wall-clock deadline.

| Order | Profile | Planning size | soft_drop | hard_drop | Lambda log step |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | floor_probe_3p7m | 3.7M | 0.05 | 0.06 | auto: 0.921034 |
| 2 | compact_6m_control | 6M | 0.015 | 0.018 | auto: 0.921034 |
| 3 | compact_6m_half_step | 6M | 0.015 | 0.018 | 0.46 |

The last two profiles have identical training configurations except for
`adaptive_lambda.log_step`. The usual controller changes lambda by a factor
of about 2.512; the 0.46 variant changes it by about 1.584. Both still adapt
lambda from validation feedback. This is a controlled test of the response
step, not fixed-lambda training. Actual physical sizes can differ between the
two profiles; compare size and validation together before interpreting test.

The completed v2 14,848,414-parameter / test 93.72% point remains the existing
large-model comparison and is not retrained or relabeled. Planning sizes in
the table are labels, not predicted measurements or pruning quotas.

The shared recipe stays LS0, min_keep_ratio 0.08, checkpoint selection only
from search epochs31-60, L1/tau0.5, entropy0, seed42, AdamW0.001/weight-decay0.0005,
fresh recovery optimizer/cosine and 200 training batches of BN calibration.
The dense-reference history and zero-epoch initializer are the same as v2.
The search/export gate implementation is unchanged. Each model stays within
150 total training epochs; completed models receive no extra training.

All three validation rows are locked before frozen official-test evaluation
on all 10,000 examples, with no training or BN updates. The comparison is
exploratory because earlier test results informed these experiments. The CSV
records `resolved_log_step` alongside the thresholds and physical parameters.

## Run from the server repository root

```bash
git fetch origin
git switch feature/resnet50-cifar10-ls0-curve7
git pull --ff-only origin feature/resnet50-cifar10-ls0-curve7

nohup .venv/bin/python -u scripts/launch_resnet50_cifar10_ls0_night9h.py \
  > resnet50_ls0_night9h_v1.log 2>&1 &
```

```bash
tail -n 100 -f resnet50_ls0_night9h_v1.log
```

The manifest is `configs/experiment/pruning_v3/resnet50_cifar10_ls0_night9h.yaml`.
New results go to `outputs/runs/resnet50_cifar10_ls0_night9h_v1/`.
`--validate-only` checks all phase configurations, the architecture floor and
the planned duration on CPU without loading data. Actual launch checks the
reference, CUDA and free space before training. Strict completed-stage resume,
checkpoint cleanup and failure continuation follow the existing curve runner.
