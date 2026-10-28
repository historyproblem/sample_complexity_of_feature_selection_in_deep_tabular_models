# AutoPruner MobileNetV2 / TinyImageNet-200 with AdamW

Repeat the five-size 20+25 experiment with AdamW and an initial learning rate
of 0.001. First place `mobilenet_baseline_partial.tar.gz` in the repository
root on the CUDA server if the original checkpoint is missing. This is the
local baseline archive; Git does not download it. Then run:

```bash
.venv/bin/python scripts/prepare_autopruner_mobilenetv2_baseline.py && \
.venv/bin/python -u src/net_complexity/tune.py \
  --config-name=tune_autopruner_mobilenetv2_tinyimagenet200_params_20_25_adamw
```

Each run starts independently from the original plain MobileNetV2 checkpoint:
`outputs/runs/20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200/checkpoints/best.pt`.
The preparation script verifies the existing file or restores just this
checkpoint from the archive and checks SHA-256
`a8230a3cc5ffd0263f8c7c6b90fc8b5a1c2c7f71207ee469b2444e74a1a5ef7a`.
It stops before tuning if inputs are missing or differ, and never replaces
another checkpoint or trains a new baseline. Use `--archive=/path/to/archive`
if the archive is stored elsewhere. The checkpoint must be present along
with the TinyImageNet data used by the previous run.
See [the baseline restoration instructions](autopruner_mobilenetv2_tinyimagenet200_45ep.md)
if the checkpoint needs to be restored. This command starts a new study;
it does not resume the completed SGD models.

The five exported sizes remain 1,000,052 / 1,300,045 / 1,599,096 /
1,900,248 / 2,200,241 parameters. Each run receives 20 search epochs and
25 fine-tuning epochs, for 45 epochs per model and 225 across the five runs.
The original shared baseline training cost is separate.

AdamW uses one parameter group, with LR 0.001 and weight decay 0.0005 for
both phases. The LR stays constant during search, then follows the existing
25-epoch cosine schedule. `model.pruning_lr` references `optimizer.lr` so
the epoch hook uses the same initial value. AdamW moments are cleared at
the search-to-fine-tuning transition, as in the previous stage reset policy.
There is no additional pretrained-backbone LR multiplier.

The recipe preserves seed 42, batch size 128, `num_workers=0`, exact keep
counts, adaptive AutoPruner lambda with an automatic log step, and search
entropy coefficient 0.3. Cross-entropy label smoothing is explicitly zero
in both phases. Once the mask is fixed, selector regularization and entropy
are disabled during fine-tuning.

Validation selects the checkpoint among epochs 21-45. The selected frozen
physical deployment is evaluated on the official TinyImageNet labeled
validation set, used as the project's test split, without optimizer or
BatchNorm updates. Comparisons to the previously inspected test results
remain exploratory.

Results are written to
`outputs/studies/<timestamp>_autopruner_mobilenetv2_tinyimagenet200_params_20_25_adamw/`.
Study and per-run names distinguish these results from the SGD series.
