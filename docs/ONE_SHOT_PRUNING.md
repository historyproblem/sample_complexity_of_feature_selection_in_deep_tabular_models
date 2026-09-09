# One-shot learned-mask pruning with fresh retraining

Branch: `feature/resnet50-one-shot-reinit`.
Protocol: `one_shot_reinit_v1`. This is deliberately separate from the existing
cyclic pruning launcher; the historical configurations are unchanged.

## What runs

1. Build or verify a **separate ordinary dense reference**: 150 continuous epochs,
   one AdamW optimizer and one cosine schedule, no pruning. Save its original
   zero-epoch random initializer and its validation accuracy at each epoch.
2. Start the gated ResNet50 from those **zero-epoch random weights**, not the
   trained dense checkpoint. Train weights and gates for **150 continuous epochs**
   with adaptive lambda and the corresponding dense validation reference.
3. Select the best search checkpoint by validation accuracy (validation CE breaks
   ties). Read each gate's raw `softmax(logits)[open]` probability and physically
   remove channels with **`p_open <= 0.5`** exactly once.
4. Keep only this compact architecture. Create **fresh random weights**, fresh
   BatchNorm affine parameters/statistics, and a new optimizer/scheduler. Train
   the compact, gate-free model from scratch for **150 continuous epochs**.
5. Select the final scratch checkpoint on validation. Evaluate its frozen
   deployment on the official 10,000-image CIFAR-10 test immediately, before the
   next predeclared job starts. Dense is also tested once, immediately after it
   is built or verified. No test result feeds training, checkpoint selection,
   adaptive lambda, or queue scheduling.

This is **300 training epochs per pruned result**, not the earlier total-150
cyclic experiment. The user's explicit 150-search + 150-reinitialization request
is the budget change for this new protocol only. A new dense reference costs
another 150 epochs once per queue. Thus the first single-job command allocates
**450 new epochs**; with a matching existing reference it allocates **300**.
The three-job queue allocates **1,050**, or **900** with a reused reference.
These epoch totals are not wall-time estimates or a promise to finish overnight.

There is no `max_param_fraction`, parameter budget, ranking-based quota, minimum
keep ratio, iterative physical recovery, or pretrained compact weight handoff.
The amount of compression is an outcome of the learned mask, not specified by a
job name. If no channel closes, the full architecture is retrained and explicitly
reported as **no compression**. If a gate closes an entire channel boundary, the
job stops with diagnostics; it does not silently reopen a channel or invent a
budget/floor to rescue the architecture.

## Configurations

The default launcher config is `configs/pruning_one_shot.yaml`; it requests only
`O1_internal_gap05_1`. `configs/pruning_one_shot_queue.yaml` requests all three:

| Job | Soft / hard validation gap vs dense | Physical pruning rule |
|---|---|---|
| `O1_internal_gap05_1` | 0.5 / 1 percentage points | Raw `p_open <= 0.5` |
| `O2_internal_gap1_2` | 1 / 2 percentage points | Same |
| `O3_internal_gap2_5` | 2 / 5 percentage points | Same |

Job definitions live in `configs/experiment/one_shot/` and inherit `_base.yaml`.
`0.005` in YAML is 0.5 percentage points, not 0.005 percentage points. The gap
thresholds control lambda, not the final mask threshold. Search probabilities
are learned during training; the filenames specify the experiment, not those
future probabilities or the final compression.

All variants keep:

- CIFAR-adapted ResNet50: `3x3` stride-1 stem, no initial max-pool, ten classes;
  internal bottleneck channel gates, unchanged external residual widths.
- Train/validation split 45,000/5,000; seed 42; batch size 128; normal train crop
  and horizontal-flip augmentation; no random validation/test augmentation.
- AdamW, learning rate `0.001`, weight decay `0.0005`, gate weight decay zero.
  A single cosine decay over 150 epochs **per phase**, with no intermediate
  optimizer resets. Scratch starts a new optimizer from the same initial LR.
- Search seed 42, fresh compact initializer seed 4242. BatchNorm state and AdamW
  state from search are discarded, not just the convolution weights.
- Adaptive lambda starts at `0.001`; warmup 10 epochs; updates every three epochs;
  three-observation accuracy window; min `1e-8`, max `80`; existing fast-pruning
  acceleration reset fix retained. Adaptive recovery of still-existing gates
  remains configured; it does not add epochs or physically prune during search.
- `ste_hard` train gates, `deterministic_hard` evaluation gates; raw mask threshold
  0.5; `initial_channels` regularization normalization. Since no physical channels
  are removed before search finishes, that denominator stays constant throughout
  the one-shot search. Scratch has no gates and therefore no lambda penalty.
- Validation accuracy as primary checkpoint monitor and validation CE as tie
  breaker. The inherited 200-epoch template is explicitly overridden to 150.

Temporary adaptive-recovery opening biases are **excluded** from physical mask
selection: the saved raw logits alone determine the architecture. The selected
search checkpoint can come from an earlier epoch even though all 150 search
epochs were consumed. Checkpoint epoch and total consumed epochs are logged
separately. The final scratch checkpoint is selected the same way.

## Why the old J1 reference cannot be reused

The old J1 ran `20 + 15 + 20 + 15 + 20 + 60` epochs, with optimizer/LR restarts.
The new search runs one uninterrupted 150-epoch cosine schedule. Its controller
must compare against a dense curve with **that same continuous schedule**.

The launcher therefore explicitly trains `D0_dense_reference` once unless
`--reference-source` names a completed compatible new reference. Validation
checks its protocol, completion, schedule, seed, model/data/optimizer signature,
zero-trained-epoch initializer, and history hashes; search checks the actual
split identity again. Old cyclic J1 results are refused, never silently used.
Reference files are used read-only; moving a run is allowed, modifying its
contents is not. Keep the reference folder while its dependent jobs run.

The common base config describes the gated search model. The dense runner
explicitly converts it to a plain all-channel model and disables the gate-only
controller; the scratch runner likewise uses a physically compact gate-free
model. This is not fixed-lambda substitution during mask search.

## Commands

Run from the repository root. Dependencies must already be installed in `.venv`.
CUDA is mandatory for these production commands; CPU fallback is rejected.

First inspect the fully resolved plan and training configs (read-only; no CUDA
check, data loading, output directory creation, or reference-file loading):

```bash
.venv/bin/python scripts/launch_one_shot_pruning.py --config-name pruning_one_shot --cfg job
```

First single experiment, including the new continuous dense reference:

```bash
.venv/bin/python scripts/launch_one_shot_pruning.py --config-name pruning_one_shot
```

All three predeclared variants, sharing one new reference:

```bash
.venv/bin/python scripts/launch_one_shot_pruning.py --config-name pruning_one_shot_queue
```

Reuse a **completed new one-shot** reference, with either its queue parent or
its `D0_dense_reference` directory (replace the example path):

```bash
.venv/bin/python scripts/launch_one_shot_pruning.py --config-name pruning_one_shot_queue \
  --reference-source outputs/runs/REPLACE_WITH_COMPLETED_ONE_SHOT_RUN/D0_dense_reference
```

Optional arguments: `--data data`, `--device cuda:0`, and `--hours 12` for an
explicit hard wall-clock cap. The default is no time cap, because the first run
contains 450 epochs and a shorter cap can interrupt it. An interrupted job is
reported as incomplete, never as a completed experiment. Existing output paths
are refused. Automatic fresh output paths are `outputs/runs/<timestamp>_<name>`.
No `nohup`, custom output flag, or backgrounding is required; logs appear live
and are also written to durable files.

The launcher first runs focused tests and a short GPU smoke through the same
search/extract/fresh-scratch path. Smoke outputs are separate diagnostics, not
experiment results, and its deliberately short epochs are not accepted by the
production configuration validator.

## Results and audit trail

```text
outputs/runs/<timestamp>_pruning_one_shot/
  launcher_config.yaml
  one_shot_queue_status.json
  provenance.json
  reference_provenance.json
  D0_dense_reference/                  # ordinary continuous dense reference
    initializer.pt                    # original random weights, trained_epochs=0
    global_history.csv
    one_shot_state.json
    deployment.pt
  O1_internal_gap05_1/
    resolved_config.yaml
    global_history.csv                # 150 search + 150 scratch = 300 rows
    one_shot_state.json               # explicit phase/total epoch counters
    search/                           # search checkpoints and training metrics
    mask_selection.json               # each channel's raw p_open and exact mask
    equivalence.json                  # physical extraction integrity check
    scratch_initializer.pt            # fresh compact weights, trained_epochs=0
    scratch/                          # independent scratch training
    deployment.pt                     # final frozen validation-selected model
  test_evaluation/
    D0_dense_reference/               # immutable per-model TEST report
    O1_internal_gap05_1/               # immutable per-model TEST report
    test_summary.json                 # aggregate TEST report, preserves partial queue
    test_comparison.csv
    test_comparison.md
```

After extraction a disposable model receives search weights solely to verify
the structural slicing implementation. **It is discarded**. The scratch model
is then built again from its fresh seed, saved with its own hash, and checked to
have fresh BN state and no gates. No transferred weight enters scratch training.

Physical parameters are counted from the compact model's actual tensor sizes;
they are not an active-parameter estimate. Test evaluation never recalibrates BN
or changes model state. Training/stage reports deliberately have no test metric;
final comparisons come from `test_evaluation`, not validation columns. Queue
failures retain completed earlier per-job test reports and checkpoints. The
current launcher does not resume an incomplete search/scratch model in place;
an explicitly reused completed dense reference is the supported reuse path.

Comparisons are single-seed and **exploratory**, since earlier test results have
already informed experiment planning. DepGraph remains the primary competitor.
