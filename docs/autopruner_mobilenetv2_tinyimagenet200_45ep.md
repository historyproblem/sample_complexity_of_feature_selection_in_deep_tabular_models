# AutoPruner on pretrained MobileNetV2 / TinyImageNet-200

This recipe loads the validation-selected plain MobileNetV2 checkpoint from
`20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200`. That run
started from torchvision `IMAGENET1K_V2` weights. The checkpoint contains the
200-class classifier as well as the backbone, so `model.pretrained_checkpoint`
must point to its project-format `checkpoints/best.pt`. It is distinct from the
`model.backbone.pretrained_weights` setting for raw torchvision weights.

From the repository root on the CUDA server, run the pilot and then all six
ratio points with one command:

```bash
.venv/bin/python -u scripts/run_autopruner_mobilenetv2_tinyimagenet200.py
```

The first run is a pilot at keep ratio 0.5 with 15 pruning epochs and 30
fine-tuning epochs. If it fails, the launcher stops before starting the sweep.
The sweep then runs keep ratios 0.3, 0.5, 0.63, 0.7, 0.8, and 0.9 in order,
with 30 pruning epochs and 15 fine-tuning epochs **per run**. Every run starts
from the same plain `best.pt` and gets its own optimizer state and 45-epoch
budget. To run only one phase, pass `--pilot-only` or `--sweep-only`; use
`--checkpoint=/absolute/path/to/best.pt` if the server keeps the baseline
elsewhere.

AutoPruner searches all 16 independently narrowable expansion widths. The
MobileNetV2 ImageNet setting in Luo and Wu's [paper](https://cs.nju.edu.cn/wujx/paper/AutoPruner_PR2020.pdf)
uses a pretrained width-1.0 backbone, a uniform channel keep ratio of 0.63,
alpha from 0.1 to 100, SGD momentum 0.9 with weight decay 0.0005 and batch
size 256, fixed learning rate 0.01 during 5 pruning epochs, then cosine decay
from 0.01 during 150 fine-tuning epochs. Their lambda starts at 10 and follows
`100 * abs(actual_keep_ratio - target_keep_ratio)`; their loss has no entropy
term. At 224x224 they obtain 71.18% ImageNet top-1 with 207.93M MACs.

This TinyImageNet-200 recipe keeps the requested 45-epoch schedules and batch
size 128. It uses the author's alpha range, SGD momentum and weight decay,
fixed search learning rate 0.01, and cosine fine-tuning learning rate. Alpha
updates every batch. The 0.63 sweep point reproduces the author's channel
target; other points explore the compression curve. The selector remains after
depthwise activation, so deployment matches the physically narrow model. A
2x2 pooled coder keeps the selectors tractable at 224x224.

The requested lambda-step and entropy changes form a hybrid objective. The
author lambda target is recomputed from each 20-batch binary-code consensus.
`model.lambda_log_step_init: auto` bounds movement toward that target at each
window; the log step is chosen to span lambda 10 to 100 within the first third
of the pruning phase. The search loss also adds `0.3 * negative_entropy`,
averaged over selectors. This is the repository's
`plus_negative_entropy` convention: it rewards uncertain soft codes while
alpha gradually binarizes them. Neither change is in the original paper.
`training_arguments.adaptive_lambda.enabled: false` refers only to the
separate Gumbel/AIG controller, which requires `model.lambda_coef`.

Validation selects the best checkpoint only from fine-tuning epochs 16–45 for
the pilot or 31–45
for the sweep. Each run writes `checkpoints/autopruner_pruned.pt` with
physically narrowed channels and evaluates that frozen deployment on the
official TinyImageNet validation set, held out as the project's test split.
Report the earlier plain baseline cost separately from each 45-epoch pruning
run; compare final quality with the official test metric and use validation
for configuration selection.
