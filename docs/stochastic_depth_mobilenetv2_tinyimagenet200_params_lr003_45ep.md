# MobileNetV2 / TinyImageNet-200 expected-active-parameter sweep

Five independent SD training runs target the linear-schedule minimum,
1.7M, 2M, 2.2M, and the full model (2,480,072 parameters, approximately 2.5M).
The targets describe the expectation of executed parameters under sampled
inference or training masks. The stored model always contains 2,480,072
parameters; individual batches can execute more or fewer than the expectation.

The exact conversion for this architecture is:

```text
expected_active_parameters = 1,604,948.8 + 875,123.2 * p_L
```

| Expected active parameters | Final survival probability p_L |
| ---: | ---: |
| 1,604,948.8 (minimum) | 0.0 |
| 1,700,000 | 0.10861464991443491 |
| 2,000,000 | 0.45142352528192603 |
| 2,200,000 | 0.6799627755269201 |
| 2,480,072 (dense) | 1.0 |

The minimum uses the existing linear schedule: early branches still have
nonzero survival probabilities, while the last eligible branch always closes
in training and sampled inference. Its parameter count remains part of the
stored model. This differs from disabling all ten SD branches, whose absolute
executed-parameter floor would be 1,402,184.

## Training

The recipe retains LR 0.03, SGD momentum 0.9, weight decay 0.0005, dropout 0.2,
linear survival probabilities, seed 42, batch 128, 45 epochs with cosine45,
and the same 224-pixel ImageNet-normalized preprocessing. Every run reloads the
original pretrained classifier-200 checkpoint and starts a fresh optimizer.
Points run sequentially from smallest to largest, one training seed each.

Run from the repository root:

```bash
.venv/bin/python -u src/net_complexity/tune.py \
  --config-name=tune_stochastic_depth_mobilenetv2_tinyimagenet200_params_lr003_45
```

The study appears at
`outputs/studies/<timestamp>_stochastic_depth_mobilenetv2_tinyimagenet200_params_lr003_45/`.
Validation selects checkpoints/configurations. Standard validation/test use
the full graph with probability-scaled residuals. BN recalibration is disabled.

## Sampled inference after training

Evaluate the selected `best.pt` checkpoints with actual branch skipping:

```bash
.venv/bin/python -u scripts/evaluate_stochastic_depth_mobilenetv2_tinyimagenet200.py \
  --study-dir outputs/studies/<timestamp>_stochastic_depth_mobilenetv2_tinyimagenet200_params_lr003_45
```

The existing evaluator accepts all five runs and their `sd_variant` tags. It
evaluates the full mode once and sampled inference five times per checkpoint,
with BN/dropout frozen and a new SD mask each batch. It reports mean executed
parameters separately from stored parameters. Since prior test comparisons
informed these experiments, describe the test results as exploratory.
