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
The sweep then runs keep ratios 0.3, 0.5, 0.6, 0.7, 0.8, and 0.9 in order,
with 30 pruning epochs and 15 fine-tuning epochs **per run**. Every run starts
from the same plain `best.pt` and gets its own optimizer state and 45-epoch
budget. To run only one phase, pass `--pilot-only` or `--sweep-only`; use
`--checkpoint=/absolute/path/to/best.pt` if the server keeps the baseline
elsewhere.

AutoPruner searches all 16 independently narrowable expansion widths. Its
regularization coefficient adapts from 20-batch consensus windows. A 2x2
max-pooled coder keeps the selectors tractable at 224x224. Validation selects
the best checkpoint only from fine-tuning epochs 16–45 for the pilot or 31–45
for the sweep. Each run writes `checkpoints/autopruner_pruned.pt` with
physically narrowed channels and evaluates that frozen deployment on the
official TinyImageNet validation set, held out as the project's test split.
Report the earlier plain baseline cost separately from each 45-epoch pruning
run; compare final quality with the official test metric and use validation
for configuration selection.
