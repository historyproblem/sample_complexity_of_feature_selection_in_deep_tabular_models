# Stochastic depth on pretrained MobileNetV2 / TinyImageNet-200

This adaptation uses the same plain 200-class MobileNetV2 checkpoint and
224x224 ImageNet-normalized input pipeline as the AutoPruner comparison.
Every run starts independently from the baseline, with fresh SGD momentum,
seed 42, batch size 128, learning rate 0.01, weight decay 0.0005 and 45 epochs
of cosine decay. The earlier baseline training cost is reported separately.

Run from the repository root on the CUDA server:

```bash
.venv/bin/python -u scripts/run_stochastic_depth_mobilenetv2_tinyimagenet200.py
```

The default checkpoint is
`outputs/runs/20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200/checkpoints/best.pt`.
Use `--checkpoint=/absolute/path/to/best.pt` for another location, or
`--batch-size=64` if batch 128 does not fit. Use one batch size throughout
the grid. The script supports `--dry-run`.

The seven ordered values are `p_L = 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0`.
The ten blocks with an identity shortcut use
`p_l = 1 - l / 10 * (1 - p_L)`; the seven transition blocks always execute.
During training one Bernoulli mask is shared by the entire batch. A closed
branch is skipped, including its BatchNorm updates and gradients. An open
branch has no inverse-probability scaling. Evaluation computes every branch
and multiplies its residual output by `p_l`, following the Huang convention.
This is a MobileNetV2 fine-tuning adaptation of stochastic depth.

The loss is cross-entropy. Stochastic-depth probabilities are prescribed,
so this method has no learned gates, entropy loss or adaptive lambda.
`p_L = 1` supplies a matched dense fine-tuning control. Every deployment has
2,480,072 parameters; stochastic depth saves training computation rather than
reducing the stored model or the standard full-graph inference cost.

Metrics record expected and observed active blocks, train-forward FLOPs,
estimated forward-plus-backward FLOPs and full inference FLOPs. FLOPs count
Conv/Linear operations as `2 * MAC`. Backward is an explicit estimate of
twice the forward cost, not a hardware measurement.

Validation selects each checkpoint and the grid winner. The official labeled
TinyImageNet validation set remains the project's test split; frozen models
are evaluated without BatchNorm recalibration. Test comparisons are
exploratory because earlier results informed this comparison. Artifacts
are stored under
`outputs/studies/<timestamp>_stochastic_depth_mobilenetv2_tinyimagenet200_pL_grid_45/`.
