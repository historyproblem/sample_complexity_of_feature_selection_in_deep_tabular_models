# MobileNetV2 v3 adaptive-lambda optimizer handoff

The MobileNetV2/TinyImageNet-200 port uses the same audited per-model budget as
the ResNet v3 handoff protocol:

- 60 epochs of shared gated search with validation-controlled adaptive lambda;
- physical export of the validation-selected learned-closed mask;
- 90 epochs of physical recovery per branch;
- 150 training epochs on every compared model path.

The generic MobileNetV2 implementation supports both pruning boundaries:

- residual output channels (`features.N.gumbel_layer`);
- local inverted-bottleneck width (`features.N.mid_gumbel_layer`), including
  non-residual blocks with an expansion convolution.

The checked-in v3 handoff profile disables the residual-output gate and searches
only the local internal width. This matches the ResNet-50 v3 profile, which sets
`gate_output: false` and searches its internal Bottleneck boundaries. It also
uses the same `paper_resnet50` gate-logit initialization; the name is historical,
but the initialization contract is architecture-independent.

Physical export slices Conv/BatchNorm tensors in original channel coordinates.
Mapped recovery applies the same slicing to AdamW moments and step counters.
Depthwise weights are mapped only on their output/group axis; the pointwise
projection is mapped on both its output and input axes. Residual-output scatter,
when used by another profile, is implemented with index-copy just like ResNet;
there is no dense selection-matrix multiplication hidden from MAC accounting.

## Server launch

Run from the repository root. A clean run first creates a separate 150-epoch
dense validation reference and the shared zero-epoch initializer, then executes
the 60+90 pruning protocol:

```bash
.venv/bin/python scripts/launch_one_shot_pruning.py \
  --config-name experiment/pruning_v3/mobilenetv2_tinyimagenet200_optimizer_handoff_60_90_repeats2 \
  --from-scratch
```

Preview all paths and compute accounting without loading data or CUDA:

```bash
.venv/bin/python scripts/launch_one_shot_pruning.py \
  --config-name experiment/pruning_v3/mobilenetv2_tinyimagenet200_optimizer_handoff_60_90_repeats2 \
  --from-scratch \
  --dry-run
```

The full handoff ablation has six 90-epoch recovery branches (three state
policies, two recovery seeds). The shared search is executed once. Therefore
the command performs 750 unique training epochs in total when the separate
dense reference is included, while every reported model comparison remains
within the required 150-epoch path budget.

Checkpoint selection and adaptive-lambda feedback use only the held-out
validation split taken from TinyImageNet-200 training images. The official
TinyImageNet-200 validation set is not constructed during training; it is used
only as the final frozen evaluation split, with no training or BatchNorm
updates. Because earlier test results may inform later experiments, those
comparisons are reported as exploratory.
