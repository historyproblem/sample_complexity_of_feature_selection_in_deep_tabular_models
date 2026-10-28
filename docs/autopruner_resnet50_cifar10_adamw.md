# AutoPruner ResNet50 / CIFAR-10 with AdamW

This reruns the original six keep ratios (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)
with AdamW and initial LR 0.001, using seed 42 and the same trained baseline.
Each model gets 32 search epochs (four stages of eight) and 118 fine-tuning
epochs, 150 in total. Baseline pretraining is a separate shared cost.

The original stage LR drops, weight decay (0.0005 during search, 0.0001 during
fine-tuning), adaptive AutoPruner regularization, batch size 256 and eight
workers are preserved. Label smoothing is explicitly zero in both phases.
AdamW moments reset after restoring the best validation state at each stage
transition. The optimizer group contains no SGD momentum or Nesterov options.

## Restore the trained baseline before launching

The required checkpoint is `best.pt` from run
`20260822_142918_resnet_baseline_cifar10`, epoch 136 of 150.
Its recorded validation accuracy is 94.1796875%; this is checkpoint metadata,
not a new test evaluation. Its SHA-256 is
`25d6f861ccb2dfc8de55420d638949f8d703ff934b3d807db408bf20874f97f5`.

Place `resnet50_cifar10_150ep_baseline_best.tar.gz` in the server repository
root. It contains a root-level `best.pt`; this archive is a local artifact,
not a file downloaded by Git. Alternatively, place the same checkpoint at
`outputs/pretrained/resnet50_cifar10_150ep/best.pt`.

Run from the repository root:

```bash
.venv/bin/python scripts/prepare_autopruner_resnet50_baseline.py && \
.venv/bin/python -u src/net_complexity/tune.py \
  --config-name=tune_autopruner_resnet50_cifar10_full_150ep_adamw
```

The preparation command verifies an existing checkpoint or restores and
verifies the archive member. If both are missing, or the hash differs, it
fails before training. It never trains a replacement baseline or overwrites
a different existing checkpoint. Other archive locations can be supplied
with `--archive=/path/to/resnet50_cifar10_150ep_baseline_best.tar.gz`.

Do not use the historical end-to-end V100 runner for this restart: without
an explicit checkpoint it trains a different 200-epoch baseline.

## Results

Study and run names end in `_adamw`, keeping the earlier SGD results separate.
Final checkpoint selection uses validation accuracy in epochs 33–150; the
selected frozen physical deployment is then evaluated on the official test
set. Comparisons with previous runs are exploratory because earlier test
results informed follow-up experiments. This configuration does not include
new training results.
