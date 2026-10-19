# ResNet-50 / CIFAR-10: physical checkpoint selection after BN calibration

Each candidate from search epochs 31-60 is now exported to its own physical
model, calibrated on 200 training batches, then evaluated on the full
5,000-example validation split. Feasible and fallback selection both use this
post-calibration validation score. Calibration uses the same seed42 training
examples and augmentations for every candidate, without gradients or optimizer
steps. The original search checkpoints and their controller states are preserved.

This addresses checkpoint selection when reopening floor-retained channels makes
the inherited BN statistics unsuitable. It does not change the gated predictor
used for adaptive-lambda feedback during search; the difference between the raw
gate mask and the floor-safe physical mask remains. Improved final test accuracy
is a hypothesis to measure, not a guaranteed consequence of this correction.

The new manifest repeats the three previous night profiles and changes the
selection procedure. Old manifests, output directories and pre-BN artifacts keep
their original meaning. Recovery rejects search artifacts from a different
selection policy; completed 150-epoch models receive no additional training.

| Order | Profile | Planning size | soft_drop | hard_drop | Lambda log step |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | floor_probe_3p7m | 3.7M | 0.05 | 0.06 | auto: 0.921034 |
| 2 | compact_6m_control | 6M | 0.015 | 0.018 | auto: 0.921034 |
| 3 | compact_6m_half_step | 6M | 0.015 | 0.018 | 0.46 |

Planning sizes are labels, not physical-parameter targets. Both step variants
retain adaptive lambda. The recipe keeps LS0, min_keep_ratio0.08, seed42,
60 search + 90 recovery epochs, fresh recovery AdamW/cosine, and the same
dense-reference history and zero-epoch initializer. Selected physical recovery
initialization is calibrated on 200 train batches as before. Selection-BN is
separately configured by `one_shot.search_selection_bn_calibration_batches` in
both phases; it is not a change to the search-export guard setting.

Thirty candidates times 200 calibration batches add 6,000 train forward batches
per model, or 18,000 for this study, plus validation and model construction.
They consume no optimizer updates and are recorded as diagnostic compute.
The old 450 training minutes plus 90-minute reserve remains a planning budget
at one minute per training epoch. The new overhead is not yet timed, so the
launcher does not promise completion within nine hours and has no hard timeout.

All validation rows are locked before frozen official-test evaluation on all
10,000 examples, with no training or BN updates on test. Comparisons remain
exploratory because previous test results informed this follow-up. CSV rows
record `search_selection_bn_calibration_batches` alongside the lambda step,
thresholds, physical parameters and final validation/test metrics.

## CPU validation

```bash
.venv/bin/python scripts/launch_resnet50_cifar10_ls0_night9h_bn_selection.py --validate-only
```

This checks both phase configurations and the physical architecture floor without
loading training data or using CUDA. Preflight prints 30 candidates, 200 batches
per candidate, the 18,000 total calibration batches and the duration caveat.

## Server launch from the repository root

```bash
nohup .venv/bin/python -u scripts/launch_resnet50_cifar10_ls0_night9h_bn_selection.py \
  > resnet50_ls0_night9h_bn_selection_v1.log 2>&1 &
```

```bash
tail -n 100 -f resnet50_ls0_night9h_bn_selection_v1.log
```

Manifest:
`configs/experiment/pruning_v3/resnet50_cifar10_ls0_night9h_bn_selection.yaml`.
Results: `outputs/runs/resnet50_cifar10_ls0_night9h_bn_selection_v1/`.
Use a new output directory; `--resume-completed` only reuses verified matching
completed stages, and never silently restarts partial training.
