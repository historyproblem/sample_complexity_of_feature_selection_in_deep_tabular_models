# Six stochastic depth runs on MobileNetV2 / TinyImageNet-200

This study compares five individual changes against one SD reference recipe.
All six runs independently load the same original plain 200-class checkpoint,
with fresh optimizer state, seed 42 and 45 fine-tuning epochs. The trained
reference run is a control result, not the initialization of the other runs.

| Run | Learning rate | Weight decay | Classifier dropout | Survival probabilities |
| --- | --- | --- | --- | --- |
| Reference | 0.01 | 0.0005 | 0.2 | Linear, final probability 0.5 |
| Lower LR | 0.003 | 0.0005 | 0.2 | Linear, final probability 0.5 |
| Higher LR | 0.03 | 0.0005 | 0.2 | Linear, final probability 0.5 |
| Lower weight decay | 0.01 | 0.0001 | 0.2 | Linear, final probability 0.5 |
| No classifier dropout | 0.01 | 0.0005 | 0.0 | Linear, final probability 0.5 |
| Uniform probabilities | 0.01 | 0.0005 | 0.2 | Uniform 0.725 for every eligible branch |

For ten eligible identity branches, linear final probability 0.5 gives
probabilities 0.95, 0.90, ..., 0.50, summing to 7.25 active branches in
expectation. Uniform 0.725 also gives 7.25. This matches expected branch
count; expected training FLOPs can differ because branch costs differ.
For a uniform schedule, `final_survival_probability` is the common probability
of every eligible branch. Seven transition blocks always execute.

The reference is already stochastic depth, not a dense `p_L=1` control.
This study measures individual changes at the reference settings, not
interactions between hyperparameters. Batch size 128, input 224x224,
ImageNet normalization, SGD momentum 0.9, no Nesterov, and cosine decay
over 45 epochs are shared. Keep each model trajectory within the project's
150-epoch total training budget, including applicable earlier training.

From the repository root on the CUDA server:

```bash
git pull --ff-only origin feature/stochastic-depth-mobilenetv2-tinyimagenet
.venv/bin/python -u src/net_complexity/tune.py --config-name=tune_stochastic_depth_mobilenetv2_tinyimagenet200_one_factor_45
```

The default checkpoint is
`outputs/runs/20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200/checkpoints/best.pt`.
To use another location, append
`model.backbone.pretrained_checkpoint=/absolute/path/to/best.pt`.
The original launch script still runs the separate seven-point `p_L` sweep;
use the command above for this six-run study.

Each checkpoint and the study winner are selected by validation accuracy.
Official test accuracy is evaluated on the selected frozen checkpoint,
without BatchNorm recalibration. Test comparisons are exploratory because
earlier results informed this experiment. Each inference model retains
2,480,072 parameters and the full computation graph.

Results are stored under
`outputs/studies/<timestamp>_stochastic_depth_mobilenetv2_tinyimagenet200_one_factor_45/`.
Each run has a distinct `sd_variant` label for result tables and plots.
