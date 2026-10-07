# AutoPruner on pretrained MobileNetV2 / TinyImageNet-200

This recipe loads the validation-selected plain MobileNetV2 checkpoint from
`20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200`. That run
started from torchvision `IMAGENET1K_V2` weights. The checkpoint contains the
200-class classifier as well as the backbone, so `model.pretrained_checkpoint`
must point to its project-format `checkpoints/best.pt`. It is distinct from the
`model.backbone.pretrained_weights` setting for raw torchvision weights.

If the baseline checkpoint is absent on the server, upload the local archive
`mobilenet_baseline_partial.tar.gz` to the repository root. It already contains
the selected checkpoint at the path expected by this recipe. Restore and verify
only that member before launching the sweep:

```bash
tar -xzf mobilenet_baseline_partial.tar.gz \
  outputs/runs/20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200/checkpoints/best.pt
echo 'a8230a3cc5ffd0263f8c7c6b90fc8b5a1c2c7f71207ee469b2444e74a1a5ef7a  outputs/runs/20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200/checkpoints/best.pt' | sha256sum -c -
```

This is the project checkpoint with a fine-tuned 200-class head; the raw
torchvision ImageNet checkpoint is not interchangeable with it.

From the repository root on the CUDA server, run the fixed-size schedule
comparison with one command:

```bash
.venv/bin/python -u scripts/run_autopruner_mobilenetv2_tinyimagenet200.py
```

If the server has only 64 MiB in `/dev/shm`, `num_workers: 0` still works:
DataLoader collates batches in the main process and does not copy them through
worker shared memory. Keep batch size 128 unless GPU or host memory becomes a
problem. To run the entire five-schedule comparison at batch size 64, use:

```bash
.venv/bin/python -u scripts/run_autopruner_mobilenetv2_tinyimagenet200.py --batch-size 64
```

Use one batch size across all five schedules because changing it changes the
number of optimizer and selector updates within each epoch.

The five runs use 5+40, 10+35, 15+30, 20+25, and 30+15 search/fine-tuning
epochs. Each run starts from the same plain `best.pt`, uses the same seed and
keep target 0.63, and receives a fresh optimizer state and a 45-epoch budget.
The comparison enables `exact_keep_count`, selecting exactly
`round(0.63 * channels)` hidden channels in each of the 16 blocks. Thus the
exported models have identical parameter and MAC counts even when the chosen
channels differ. At width 1.0 and 200 classes the fixed deployment has
1,811,897 parameters (4,477 of 7,104 gated channels kept). This fixed-width
rule is an experimental control, not an
operation in the original paper. Use `--checkpoint=/absolute/path/to/best.pt`
if the server keeps the baseline elsewhere.

AutoPruner searches all 16 independently narrowable expansion widths. The
MobileNetV2 ImageNet setting in Luo and Wu's [paper](https://cs.nju.edu.cn/wujx/paper/AutoPruner_PR2020.pdf)
uses a pretrained width-1.0 backbone, a uniform channel keep ratio of 0.63,
alpha from 0.1 to 100, SGD momentum 0.9 with weight decay 0.0005 and batch
size 256, fixed learning rate 0.01 during 5 pruning epochs, then cosine decay
from 0.01 during 150 fine-tuning epochs. Their lambda starts at 10 and follows
`100 * abs(actual_keep_ratio - target_keep_ratio)`; their loss has no entropy
term. At 224x224 they obtain 71.18% ImageNet top-1 with 207.93M MACs.

This TinyImageNet-200 recipe keeps the requested 45-epoch total and batch size
128. It loads batches with `num_workers: 0` because worker-side collation of
224x224 images exhausted shared memory on the CUDA server. It uses the
author's alpha range, SGD momentum and weight decay,
fixed search learning rate 0.01, and cosine fine-tuning learning rate. Alpha
updates every batch. [TinyImageNet has 100,000 training images](https://cs231n.stanford.edu/reports/2016/pdfs/405_Report.pdf);
with this recipe's 10% validation split, about 90,000 remain for training, or
about 703 full batches per epoch at batch size 128.
Thus 5 TinyImageNet search epochs give about 3,515 updates, compared with
about 25,000 for 5 ImageNet epochs at batch size 256. The schedule series
tests whether more search updates improve the final validation accuracy enough
to justify fewer fine-tuning epochs. The selector remains after depthwise activation,
so deployment matches the physically narrow model. A
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

For each schedule, validation selects the best checkpoint only from its
fine-tuning phase. The study chooses the schedule with highest validation
accuracy and records it in `best_trial.yaml`. Each run writes
`checkpoints/autopruner_pruned.pt` with physically narrowed channels and
evaluates that frozen deployment on the official TinyImageNet validation set,
held out as the project's test split. Use validation to choose the schedule;
report official test results as exploratory because they can inform subsequent
experiments. Once the schedule is chosen, the separate keep-ratio sweep can use
that schedule. Report the earlier plain baseline cost separately from each
45-epoch pruning run.
