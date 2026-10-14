# MobileNetV2 / TinyImageNet-200 p_L sweep at learning rate 0.03

This recipe runs the same six final survival probabilities as the historical
ResNet50 / CIFAR-10 FLOPs study, in order: `1.0, 0.9, 0.8, 0.7, 0.5, 0.3`.
The first run is the dense control. Each point starts independently from the
same original pretrained MobileNetV2 checkpoint and a fresh SGD optimizer.

The remaining settings match the baseline of the latest one-factor SD study:
45 fine-tuning epochs, seed 42, batch 128, weight decay 0.0005, momentum 0.9,
classifier dropout 0.2, linear survival schedule, cosine scheduler with
`T_max=45`, and the same 224-pixel ImageNet-normalized TinyImageNet-200 pipeline.
The default checkpoint remains
`outputs/runs/20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200/checkpoints/best.pt`.

## Training

Run from the repository root on the server:

```bash
.venv/bin/python -u src/net_complexity/tune.py \
  --config-name=tune_stochastic_depth_mobilenetv2_tinyimagenet200_pL_grid_lr003_45
```

The study is written under
`outputs/studies/<timestamp>_stochastic_depth_mobilenetv2_tinyimagenet200_pL_grid_lr003_45/`.
There are six sequential runs, one training seed per point, with no trial pruning.
Checkpoints and configurations are selected by validation accuracy. Standard
validation and test use the full graph with survival-probability scaling.
BatchNorm recalibration is disabled. This command starts new runs, each within
the existing 45-epoch budget; it does not continue previous SD checkpoints.

## Sampled inference after training

To evaluate actual branch skipping for the resulting saved `best.pt` files,
pass the new study directory to the existing evaluator:

```bash
.venv/bin/python -u scripts/evaluate_stochastic_depth_mobilenetv2_tinyimagenet200.py \
  --study-dir outputs/studies/<timestamp>_stochastic_depth_mobilenetv2_tinyimagenet200_pL_grid_lr003_45
```

The evaluator performs one standard test pass and five sampled test passes per
checkpoint. Sampled inference resamples residual-branch masks per batch while
keeping BatchNorm and classifier dropout in eval mode. It reports executed
parameters separately from the full stored parameter count, 2,480,072.
These test comparisons are exploratory because previous test results have
already informed the experiments.
