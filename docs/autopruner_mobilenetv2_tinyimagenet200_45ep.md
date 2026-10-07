# AutoPruner on pretrained MobileNetV2 / TinyImageNet-200

This recipe loads the validation-selected plain MobileNetV2 checkpoint from
`20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200`. That run
started from torchvision `IMAGENET1K_V2` weights. The checkpoint contains the
200-class classifier as well as the backbone, so `model.pretrained_checkpoint`
must point to its project-format `checkpoints/best.pt`. It is distinct from the
`model.backbone.pretrained_weights` setting for raw torchvision weights.

From the repository root on the CUDA server:

```bash
.venv/bin/python -u src/net_complexity/train.py \
  experiment=autopruner_mobilenetv2_tinyimagenet200_45ep \
  model.pretrained_checkpoint=outputs/runs/20261002_194727_baseline_pretrained_mobilenetv2_tinyimagenet200/checkpoints/best.pt
```

The 45 new training epochs are 15 epochs of AutoPruner search across all 16
independently narrowable expansion widths, then 30 epochs of fine-tuning with
fixed binary masks. The regularization coefficient adapts from the method's
20-batch consensus windows. A 2x2 max-pooled coder keeps the MobileNetV2
selectors tractable at 224x224. The target channel keep ratio is 0.5.

Validation selects the best checkpoint only from epochs 16–45. The run writes
`checkpoints/autopruner_pruned.pt` with physically narrowed channels and
evaluates that frozen deployment on the official TinyImageNet validation-as-test
split. The earlier plain baseline cost should be reported separately from the
45-epoch pruning run.
