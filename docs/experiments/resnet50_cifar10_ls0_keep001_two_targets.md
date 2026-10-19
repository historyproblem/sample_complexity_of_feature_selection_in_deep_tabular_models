# ResNet-50 / CIFAR-10: two follow-ups with the old channel floor

The user requested two configurations near 4.5M and 14M parameters and restoring
the floor used by the old red-circle LS0 series. That floor was
`min_keep_ratio=0.001` (0.1%), not 0.01. It is a minimum surviving fraction at
each prunable boundary; the gate probability threshold remains 0.5.

| Planning size | soft_drop | hard_drop | Lambda step |
| ---: | ---: | ---: | --- |
| 4.5M | 0.015 | 0.018 | auto: 0.921034 |
| 14M | 0.00025 | 0.0005 | auto: 0.921034 |

These are quality-threshold probes, not parameter-budget enforcement. The
starting profiles previously yielded 4.738816M / validation93.52% and
14.848414M / validation94.20% with floor0.08. At the old floor0.001 and old
pre-BN selection, the second thresholds yielded4.659889M. The restored floor
and corrected selection procedure can change architecture substantially;
actual sizes must be measured before claiming coverage near either target.
The thresholds are starting points based on prior validation and sizes,
not selected using the test results.

Both follow-ups retain label smoothing0, adaptive lambda with its automatic
step, seed42 and the same reference/zero-epoch initializer, L1/tau0.5,
60 search plus90 recovery epochs, fresh recovery AdamW/cosine and BN200.
For every candidate in epochs31-60, a disposable physical model is calibrated
on the same200 train batches before full validation. Test is frozen and
evaluated only after validation selection; comparisons remain exploratory.
The difference between the gated lambda-feedback mask and the floor-safe
physical mask is not removed by lowering this floor.

Existing floor0.08 manifests and results retain their original meaning.
Each new configuration has its own study/output/test paths. Recovery and resume
reject mismatched floors, selection windows, BN policies or reference inputs.
Each new model remains within150 training epochs, without extending prior runs.

## Check each configuration on CPU

```bash
.venv/bin/python scripts/launch_resnet50_cifar10_ls0_curve7.py \
  --config configs/experiment/pruning_v3/resnet50_cifar10_ls0_keep001_target4p5m.yaml \
  --validate-only
.venv/bin/python scripts/launch_resnet50_cifar10_ls0_curve7.py \
  --config configs/experiment/pruning_v3/resnet50_cifar10_ls0_keep001_target14m.yaml \
  --validate-only
```

## Launch separately from the server repository root

Run each after the current training queue finishes; do not run both commands
concurrently on the same training GPU. The 14M-labelled probe can go first.

```bash
nohup .venv/bin/python -u scripts/launch_resnet50_cifar10_ls0_curve7.py \
  --config configs/experiment/pruning_v3/resnet50_cifar10_ls0_keep001_target14m.yaml \
  > resnet50_ls0_keep001_target14m_v1.log 2>&1 &
```

```bash
nohup .venv/bin/python -u scripts/launch_resnet50_cifar10_ls0_curve7.py \
  --config configs/experiment/pruning_v3/resnet50_cifar10_ls0_keep001_target4p5m.yaml \
  > resnet50_ls0_keep001_target4p5m_v1.log 2>&1 &
```
