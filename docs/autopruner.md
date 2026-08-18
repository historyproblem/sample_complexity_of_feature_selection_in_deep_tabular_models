# AutoPruner (Luo & Wu, 2018)

This implementation follows [the AutoPruner paper](https://arxiv.org/abs/1805.08941)
and the [authors' official repository](https://github.com/Roll920/AutoPruner), pinned
for comparison to commit
[`4618a775`](https://github.com/Roll920/AutoPruner/tree/4618a775bbf48d1166012b9b671f84c71b114e26).
It is a device-agnostic reimplementation; no source file was copied verbatim.

## What is implemented

For an activation tensor `N x C x H x W`, each selector:

1. averages activations over the minibatch, producing `1 x C x H x W`;
2. applies `2 x 2` max pooling (`1 x 1` in the final 7x7 ImageNet stage);
3. applies a learned full-spatial convolution equivalent to a
   `C x (C H' W')` fully connected coder;
4. computes `v = sigmoid(alpha * coder(x))`;
5. multiplies every sample's channels by the same vector `v`.

The objective for every selector in the active ResNet group is

```text
cross_entropy + lambda * (mean(abs(v)) - target_keep_ratio)^2
```

The selector is deterministic, but input-dependent during pruning. Every 20
batches, channel values are thresholded at 0.5 and combined by majority vote.
Validation and subsequent pruning groups use that static binary consensus.
The exporter removes the selectors and slices `conv1`, `bn1`, `conv2`, `bn2`,
and the input dimension of `conv3`, so reported deployment MACs are real rather
than idealized masked-network MACs.

The ResNet-50 placement matches the authors' code: selectors follow the first
and second ReLU of every bottleneck, except the last bottleneck in stage 4.
Stages 1 through 4 are pruned sequentially. The best validation state of each
stage is restored before the next stage, and SGD state is reset, matching the
authors' four-script workflow. Earlier stages are represented with hard masks
during search and become physically narrow in the exported model; these forms
have the same forward function.

## Author-sourced hyperparameters

The following values come from the official ResNet-50 implementation, chiefly
[`main.py`](https://github.com/Roll920/AutoPruner/blob/4618a775bbf48d1166012b9b671f84c71b114e26/ResNet50/50/main.py)
and
[`fine_tune_compressed_model.py`](https://github.com/Roll920/AutoPruner/blob/4618a775bbf48d1166012b9b671f84c71b114e26/ResNet50/50/fine_tune_compressed_model.py):

| Setting | Value |
| --- | ---: |
| target keep ratios evaluated by the paper | `0.5`, `0.3` |
| pruning epochs per ResNet stage | `8` |
| final fine-tuning epochs | `30` |
| sigmoid `alpha` | linear `1 -> 100`, updated every `100` batches |
| binary consensus window | `20` batches |
| initial regularization coefficient | `10` |
| adaptive coefficient | `100 * abs(binary_keep_ratio - target_keep_ratio)` |
| initial pruning threshold / decrement | `95%` / `5` percentage points |
| polarization test | at least `90%` outside `[0.2, 0.8]` |
| pruning SGD | LR `1e-3`, momentum `0.9`, weight decay `5e-4` |
| pruning LR drop | `10x` after epoch 4 of each 8-epoch stage |
| final fine-tuning SGD | LR `1e-3`, momentum `0.9`, weight decay `1e-4` |
| final LR drops | `10x` every 10 epochs |
| coder weight initialization | Normal(`0`, `10 * sqrt(2 / (C H' W'))`) |

The paper's ResNet-50 experiment starts from a pretrained network. Accordingly,
the experiment config has `require_pretrained: true` and fails early if no
checkpoint is supplied.

## Project experiment and explicit adaptation

The ready-to-run recipe is a CIFAR-10 adaptation because this repository's
controlled comparison protocol uses CIFAR-10. It keeps every method-specific
setting above, but necessarily changes the paper's data/model interface:

- CIFAR-10, 32x32 inputs and 10 classes instead of ImageNet, 224x224 and 1000;
- a 3x3 stride-1 stem without max pooling instead of the ImageNet 7x7 stem;
- the shared project batch size 128 instead of the authors' ImageNet batch 256.

Supply a trained project ResNet-50 checkpoint:

```bash
python3 src/net_complexity/train.py \
  experiment=autopruner_resnet50_cifar10 \
  model.pretrained_checkpoint=/absolute/path/to/checkpoints/best.pt
```

The loader accepts a project run checkpoint or a raw state dict. Biases from
the project's legacy biased bottleneck convolutions are folded into BatchNorm
running means, preserving their evaluation function in the paper's bias-free
ResNet bottlenecks.

Run the two author-reported keep ratios as a grid:

```bash
python3 src/net_complexity/tune.py \
  experiment=autopruner_resnet50_cifar10 \
  tuning=autopruner_keep_ratio_grid \
  model.pretrained_checkpoint=/absolute/path/to/checkpoints/best.pt
```

To export a trained model programmatically:

```python
from net_complexity.models.autopruner import save_pruned_autopruner_checkpoint

save_pruned_autopruner_checkpoint(model, "autopruner_pruned.pt")
```

Test metrics include temporary coder parameters plus dense/pruned deployment
parameters, MACs, FLOPs, and their actual reductions. A normal training run
also writes the physical artifact to
`checkpoints/autopruner_pruned.pt` after restoring the best fine-tuning model.
