# Ten stronger MobileNet pruning runs

Ten new independent MobileNetV2/TinyImageNet-200 models start from ImageNet1K V2.
Every model performs search, physical channel removal with inherited trained
weights, recovery, and official frozen test. Search+recovery always totals45
training epochs,450epochs across the grid. No historical run is extended.

The first six runs cross three initial lambda values with the two existing
gate-initialization modes. Runs7-8 compare quality bands with run3. Runs9-10
compare timing/phase allocation with run3; automatic step also changes, so these
are not isolated warmup effects. These are pressure probes, not target-size
constraints; actual sizes may be non-monotonic or cluster together.

| ID | lambda_init | Nominal p_open | soft/hard, pp | Lambda warmup | Search/recovery | Auto update factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r01_p87_l1em2 |0.01 |0.87 |10/20 |5 |30/15 |1.318257 |
| r02_p98_l1em2 |0.01 |0.98 |10/20 |5 |30/15 |1.318257 |
| r03_p87_l1em1 |0.1 |0.87 |10/20 |5 |30/15 |1.202264 |
| r04_p98_l1em1 |0.1 |0.98 |10/20 |5 |30/15 |1.202264 |
| r05_p87_l1 |1.0 |0.87 |10/20 |5 |30/15 |1.096478 |
| r06_p98_l1 |1.0 |0.98 |10/20 |5 |30/15 |1.096478 |
| r07_quality_band |0.1 |0.87 |4/10 |5 |30/15 |1.202264 |
| r08_wide_band |0.1 |0.87 |15/25 |5 |30/15 |1.202264 |
| r09_search35 |0.1 |0.87 |10/20 |3 |35/10 |1.154782 |
| r10_recovery20 |0.1 |0.87 |10/20 |8 |25/20 |1.311134 |

## Authorized parameter changes

- `model.lambda_coef`: initiallambda, still adaptive throughout search.
- `model.gumbel_init_mode`: `paper` gives nominal p_open0.86989;
  `paper_resnet50` gives0.98016. Existing initialization noise is unchanged.
- `training_arguments.adaptive_lambda.soft_drop` and `hard_drop`: fractions of
  accuracy, e.g.0.10/0.20 means10/20percentage points.
- `training_arguments.adaptive_lambda.warmup_epochs`: the initial lambda stays
  active and fixed during warmup; Gumbel sampling is already enabled.
- `cyclic_channel_pruning.gumbel_epochs`, `recovery_epochs`, `final_epochs`:
  search/recovery allocation. Stage `num_epochs` and cosine `T_max` follow these
  lengths automatically; learning-rate values are unchanged.

The removal threshold is fixed at `cyclic_channel_pruning.g_prob_threshold=0.9`
as requested. Search forward/validation still use the existing binary threshold
`model.backbone.block.gate_threshold=0.5`. All runs use stochastic `gumbel_hard`
training with beta1/temperature1 and deterministic-hard validation. Final
reported test evaluates the physically compact, gate-free recovery model.

The lambda step is always literal `auto`, with target fraction1 and no boost:
`log_step = log(10/lambda_init)/(search_epochs-warmup_epochs)`. The existing
minimum step is retained; it does not clamp any of these ten configurations.
Lambda bounds1e-8..100, no parameter-removal quota, internal/output gate semantics,
minimum one channel per group, LS0/entropy0, seed42, width1/classifier200,
data/split/augmentation, AdamW, LR1e-3 and pretrained LR1e-4, weight decay5e-4,
batch128/workers0 and all other model/training settings stay fixed.

Actual configs are compared against the open98 Gumbel recipe with only the
authorized method overrides and technical output labels. Best-validation search
mask and transferred weights come from the same best.pt; selected controller
state is held. Recovery has no gates or regularization, uses the same existing
fresh-optimizer policy, and chooses its frozen deployment on validation.

## Reference and saved results

The existing historical baseline archive is imported; no baseline is fitted.
Its full86-epoch history is retained. The importer keeps its historical1-30
selection metadata, while this grid separately records actual controller
coverage1-35 for the longest search. The historical baseline's150-epoch cosine
recipe remains explicitly identified, and its weights do not initialize searches.

Output: `outputs/runs/mobilenet_strong10_20261128/<id>`.
`comparison.csv`/`comparison.json` update after each completed full model and
pair actual compact parameters with final physical test accuracy. Search test
values are not used as compact deployment quality. Each search and recovery
retains checkpoints/best.pt and checkpoints/last.pt; there is no checkpoint cleanup.
Keep those weights when backing up, not just the metadata.

The sequential launcher validates every existing run before starting any new
one. Matching complete models are reused. Partial, changed or foreign results
stop the queue without retraining. Source/config/reference fingerprints are
rechecked between models, and the output lock prevents two simultaneous launches.

## Server command

Run from the repository root, using the standard requirements file. The existing
baseline archive must be present at outputs/baselines/mobilenet_baseline_partial.tar.gz
or passed with `--baseline-archive PATH`. No old workflow is resumed.

```sh
if git fetch origin &&
   git switch feature/mobilenet-strong-pruning-grid10 &&
   git pull --ff-only &&
   (test -x .venv/bin/python || python3 -m venv .venv) &&
   .venv/bin/python -m pip install --no-cache-dir -r requirements.txt &&
   .venv/bin/python scripts/run_mobilenet_strong10.py --inspect; then
  mkdir -p outputs/logs
  nohup .venv/bin/python -u scripts/run_mobilenet_strong10.py \
    > outputs/logs/mobilenet_strong10_20261128.log 2>&1 < /dev/null &
  tail -f outputs/logs/mobilenet_strong10_20261128.log
fi
```

`--dry-run` prints the reference-import command and ten full-model commands
without third-party imports or training. `--inspect` verifies configs, inputs,
data and CUDA without starting model training. Test comparisons are exploratory
because prior results informed this sweep; settings/checkpoints use validation.
