# Frozen inference with actual stochastic-depth branch skipping

Run the existing validation-selected checkpoints with train-style depth masks.
Each eligible residual branch samples one independent Bernoulli per batch.
A closed branch is not executed. An open branch contributes its complete
output, with no multiplication by its survival probability and no inverse
probability scaling. The surrounding model remains in eval mode: classifier
dropout is off and BatchNorm uses its saved statistics without updates.
Parameters and persistent buffers are hashed before and after every pass.
No optimizer, training, BN recalibration or new checkpoint selection runs.

This is a custom stochastic inference variant, distinct from Huang's standard
full-graph, probability-scaled expected inference. All parameters remain stored
in the checkpoint. Mean active parameters describe executed branches; this
does not export one fixed physically smaller model. Masks change per batch.
Results depend on batch size, which is recorded and shared across all runs.

From the repository root, on the server holding the original checkpoints:

```bash
.venv/bin/python -u scripts/evaluate_stochastic_depth_mobilenetv2_tinyimagenet200.py \
  --study-dir outputs/studies/20261012_135959_stochastic_depth_mobilenetv2_tinyimagenet200_one_factor_45
```

The downloaded results archive excludes weights and cannot be used for this
evaluation. The script requires every original `runs/*/checkpoints/best.pt`
and checks its epoch against saved validation selection. `--check-only`
checks paths and selection metadata without loading weights or data.
Use `--data /absolute/path/to/tiny-imagenet-200` if the dataset moved.
The default device is cuda:0, num_workers=0, batch size comes from the run
config (128 in this study). `--batch-size` overrides it for every run.

For each of the six frozen checkpoints, the script evaluates standard expected
inference once and stochastic inference five times with seeds 42 through 46.
`--repeats` and `--seed` change that predefined evaluation protocol. Each image
receives one prediction per pass. Repeats report mean accuracy and sample
standard deviation, not an ensemble of predictions or a best seed on test.
Test comparisons remain exploratory.

Accuracy is `total_correct / total_examples`, including the short final batch.
Cross-entropy and active block/parameter counts are also sample-weighted.
FLOPs use the existing Conv/Linear operation collector (2*MAC plus bias adds),
with actual sampled masks, not just the theoretical expected active count.
Weights, normalization and survival probabilities come from each saved run.

Outputs are written to a new directory:

```text
<study>/stochastic_inference/<UTC_timestamp>/summary.json
<study>/stochastic_inference/<UTC_timestamp>/comparison.csv
```

Source run summaries and checkpoints are not overwritten. The new report
includes integer correct/example counts, each repeat's accuracy and compute,
checkpoint hashes and persistent-state hashes. It also records that original
checkpoint selection used the legacy unweighted mean of batch accuracies.
Correcting inference accuracy does not reconstruct the sample-weighted best
checkpoint among all 45 historical epochs. The shared `Accuracy` metric is
fixed for future training/evaluation runs; old logs remain unchanged.
