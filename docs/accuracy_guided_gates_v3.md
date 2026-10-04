# Accuracy-guided gates v3: repository contract

`accuracy_guided_gates_v3` is an explicit opt-in repository protocol. The existing
fixed pilot, `adaptive_lambda_v1` (pilot v2), parameter-budget launchers, results,
and strict validators retain their historical meanings. They are not relabelled
as v3. No new baseline, DepGraph comparison, reset/reopen policy, seed grid, sweep,
or experiment queue is introduced here. The experiment programme remains pending.

## Current code and compatibility profile

The implementation reuses the shared `training/engine.py`, `initial_channels`
gate regularization, `channel_pruning.py` transfer/export, the Bottleneck physical
model, and `pruning_measurement.py` physical counting. The new orchestration is
`training/accuracy_guided_pruning.py`; it calls `select_learned_closed`, not the
historical `select_by_budget` selector. The new launcher does not import or invoke
the historical A1/A2/A3/A4 job queue.

The inherited current profile is exactly:

```
S20 -> P -> R15 -> S20 -> P -> R15 -> S20 -> P -> R60 = 150 training epochs
```

Each P is an explicit zero-epoch commit attempt. A no-op is legal and does not
cancel recovery or the next search. Every training stage restarts AdamW and cosine
scheduling. Surviving Conv/BN tensors persist; this is weight preservation, not
optimizer-state preservation. Search uses internal Bottleneck channel gates with
threshold 0.5 and min_keep_ratio 0.5. The base configuration preserves seed/split
42, CIFAR10 45,000/5,000 split, batch 128, the existing ResNet50 architecture,
paper_resnet50 gate initialization, augmentations, AdamW LR 0.001 and ordinary
weight decay 0.0005. Gate weight decay and entropy coefficient are zero.

The profile reads the existing dense reference at
`outputs/runs/20260907_134446_507828_pruning_dense_control/J1_dense_control/` and its
sibling `shared_random_seed42.pt`. The initializer must have `trained_epochs=0`,
matching seed and tensor hash. Trained dense deployment weights are never used to
initialize a new run. Reference creation has its own external training cost.
Missing files block execution; the launcher never trains a replacement reference.
Input checks compare the validation source configuration, split, weighted accuracy
metric, full epoch coverage and initializer provenance. A synthetic smoke uses
separate synthetic reference/initializer files and cannot access official test.
Its reference is explicitly marked `synthetic_reference=true`: the CSV contains
programmed feedback at logical epochs, with zero actual dense-reference training.
This fixture is refused by the full profile and makes no measured-quality claim.

## Loss and meaning of lambda

For boundary b, `n_b0` is its original width, `n_bt` counts permanent survivors,
`r_b=n_bt/n_b0`, and `M0` is the original number of boundaries. RAW `p_bj` precedes
runtime opening bias and permanent masking. The actual canonical training loss is

```
L_gate = alpha / M0 * sum_b [sum_j(m_bj * p_raw_bj) / n_b0]
       = 1 / M0 * sum_b [lambda_effective_b * mean_survivor_p_raw_b]
lambda_effective_b = alpha * n_bt / n_b0
```

`alpha` is the single quality-controlled scalar (`lambda_base`). No survivor
ratio is multiplied into alpha again. For 100 -> 50 channels at alpha 0.001,
effective lambda becomes 0.0005 and the coefficient per survivor remains 0.00001
before the unchanged `1/M0` factor. An empty boundary contributes zero without
0/0 and remains in M0. Empty-boundary readouts are null/status; structurally invalid
exports are rejected separately. Original ids, widths and M0 are checkpointed and
validated, including on transfer. Permanently disabled coordinates have no task
or regularization gradient. Bypass and ungated recovery have zero gate penalty.

Diagnostics distinguish raw/effective survivor zero probability, survivor hard
closure, permanent physical removal and original-coordinate zero mass. Aggregation
is weighted by channel count. Old `average_zero_prob` keys retain their legacy
meaning and must not be relabelled survivor metrics. Removing low-p coordinates
can legitimately change the survivor mean through composition; disabled channels
do not contribute an artificial off=1 to the survivor statistic.

## Accuracy-only controller and clocks

The base settings are technical defaults, not tuned outcomes:

| Parameter | Value |
|---|---:|
| alpha_init / alpha_min / alpha_max | 0.001 / 1e-8 / 80 |
| soft_drop / hard_drop | 0.02 / 0.05 accuracy units |
| gap_window / update_every_search_epochs | 3 / 3 |
| initial_search_warmup / reentry_samples | 10 / 3 |
| log_step | log(2) |

The paired observation is `A_ref(global_training_epoch)-A_validation`. The
controller smooths these gaps. At an allowed update, mean gap <=0.02 doubles alpha,
gap in (0.02,0.05] holds alpha, and gap >0.05 halves it, subject to alpha bounds.
The drops are 2 and 5 percentage points, not percentages relative to reference.
Feedback strives to retain quality; it cannot guarantee the tolerance at each
step or guarantee a feasible final deployment.

`global_training_epoch` counts all completed training epochs, including rejected
candidate recovery and fallback. `search_epochs_consumed` counts gated search;
`local_search_epoch` counts the current continuous search segment. The first
segment updates at local 13,16,19,...; after physical recovery, the next segment
updates at local 3,6,9,... using three new observations. Physical recovery holds
alpha and does not grow search clocks or feedback windows. Missing or nonfinite
reference/feedback causes an explicit error; it never means increase alpha.

After recovery, surviving raw gate logits are carried and selected physical
Conv/BN weights are loaded. An explicit, idempotent transition/rebase preserves
alpha and consumed clocks and clears old forward-mode observations. Effective
lambdas are recomputed from original ids and the permanent mask. No reset_open,
reopen_preserve_rank, added warmup, gate revival or alpha decay is performed.
The actual first forward is checked after `apply_initial_state`. Diagnostics
compare physical, all-open carrier and carry-gated carrier on the same validation
examples without gradients or BN changes and with isolated RNG streams.

## Learned commit, selection and recovery

Only previously surviving channels that are closed under both RAW eligibility and
the actual effective hard forward may be proposed. The module supplies its hard
threshold/tie semantics. Existing floors/dependencies deterministically block
unsafe removals and all ids remain original coordinates. There is no top-k fill,
target remaining ratio, compulsory percentage or MAC/parameter quota. Examples:
all probabilities 0.9 produce zero removals; two eligible hard-closed coordinates
produce only those two removals if floors allow them. A raw-closed coordinate held
open by effective runtime state cannot be exported as a learned closure.

Every search checkpoint is compared with the same reference at that search's end
epoch. Quality feasibility is primary; smallest admissible physical parameter
count is secondary, followed by validation accuracy, CE and earlier epoch. If no
search checkpoint is feasible, no new deletion is committed. Checkpoint snapshots
include the evaluation runtime, alpha used and next alpha, controller state and
clocks. Selecting an early checkpoint does not rewind consumed training cost.

Physical transfer equivalence is checked before calibration. Opening blocked closed
survivors is reported separately from transfer mismatch. Immediate surgery quality
is diagnostic; scheduled recovery still happens. The selected recovery checkpoint
must reach `A_ref(recovery_end)-hard_drop`. No extra parent-relative 8%/1% guards
are silently applied. A quality failure rejects the provisional topology, restores
the last accepted physical model/mask, stops further commits, and uses remaining
epochs for ungated fallback. If none was accepted, fallback is the all-open physical
parent exported from the current search, with that origin recorded. Rejected work
stays charged. Final feasibility uses the reference at the actually completed budget;
failure is labelled `infeasible`.

## Artifacts, migration and limits

`protocol_state.json` records the stage plan, selected checkpoints, commit decisions,
transitions, epoch events, consumed ledger and provenance. `deployment.pt` is an
explicit `physical_ungated` artifact. Full-width masked search carriers are not
reported as compact convolutions. Reports separate physical parameter/buffer
counts, measured forward Conv/Linear MACs, actual optimizer updates, consumed
training examples, wall time, and diagnostic/calibration overhead. Equal epochs
do not imply equal training FLOPs; no heuristic backward count is called a
measurement. CUDA memory/latency are not claimed from CPU smoke tests.

Epoch snapshots are written atomically and support `torch.load(weights_only=True)`.
Evaluation runtime and continuation runtime are separate: a checkpoint with nonzero
legacy open_bias must reproduce its evaluated predictor, not its next controller
state. `checkpoint_runtime_status` labels old snapshots without complete runtime
metadata as `legacy_incomplete_runtime`. Historical artifacts remain readable by
their historical paths; incompleteness is not disguised as exact adaptive resume.

`migrate_legacy_controller_to_accuracy_only` is an explicit handoff migration of a
complete, inactive legacy controller snapshot. It preserves alpha and rebases the
window; active anti-collapse/open-bias episodes are refused. This migration is
labelled `migrated_handoff_not_exact_resume` and does not claim legacy run replay.

The shared engine has exact **epoch-boundary, same-stage** resume that restores
optimizer, scheduler, RNG, runtime, masks and consumed counters and checks stage
provenance. Exact mid-search resume does not rebase the feedback window. Partial
epoch snapshots and persistent DataLoader worker RNG are refused. **Whole-plan
transaction resume is not implemented**: `--resume-from` explicitly fails rather
than restarting under the existing run id or guessing a rollback transition.
This remains an integration limitation. Re-running a completed training budget
with extra recovery is not a supported way around it.

## Verification commands

Run from the repository root; do not prepend a directory change. The dry-run reads
configuration and existing input metadata without training, CUDA initialization,
CIFAR download or output-directory creation:

```sh
.venv/bin/python scripts/launch_accuracy_guided_pruning.py --dry-run
.venv/bin/python -m pytest -q tests/test_accuracy_guided_gate_contract.py tests/test_accuracy_only_runtime.py tests/test_accuracy_guided_config.py tests/test_accuracy_guided_iterative.py tests/test_accuracy_guided_evaluator.py
.venv/bin/python scripts/smoke_accuracy_guided_pruning.py --output /tmp/accuracy-guided-v3-smoke
```

Use a fresh smoke output directory. The smoke is CPU-only with a tiny-width real
Bottleneck backbone and eight synthetic epochs, exercising six search epochs,
two physical recovery epochs, carry/rebase and learned commit followed by no-op.
The integration tests separately cover quality rejection/fallback and technical
failure. These checks establish code/runtime behavior, not CIFAR10 accuracy.

The independent final evaluator checks the frozen selected **physical** deployment.
`--check-only` performs CPU artifact validation with no official test access:

```sh
.venv/bin/python scripts/evaluate_pruning_test.py --protocol-v3 --run-dir PATH_TO_COMPLETED_V3_RUN --check-only
.venv/bin/python scripts/evaluate_pruning_test.py --protocol-v3 --run-dir PATH_TO_COMPLETED_V3_RUN --data data --device cpu --output PATH_TO_FRESH_TEST_REPORT
```

The second command is for separately requested evaluation after an eligible full
run; official test is not run as part of this repository change. The evaluator
does no training, selection or calibration and verifies unchanged weight/BN hashes.
Synthetic and gated-only artifacts are refused. Missing test accuracy remains
missing; validation is not substituted. Test-informed later comparisons are
exploratory, not a claim of an untouched holdout.

## Verification environment and unperformed checks

The available local interpreter is Python 3.9.6 with torch 2.8.0. This differs
from the repository's declared Python >=3.10 and pinned torch 2.10.0 /
torchvision 0.25.0 environment. The passing local checks are evidence for this
actual environment only; the declared server dependency set has not been tested
here. No fake imports or stand-in dependency modules were installed.

The config, new frozen evaluator and historical frozen evaluator tests passed
together: 103 tests. The separate tiny real-Bottleneck iterative suite and
standalone eight-epoch synthetic smoke were also executed; their artifacts retain
the actual consumed ledger. Base dry-run returned 150 epochs and explicitly
reported the four missing reference/initializer paths. Running the independent
final evaluator's `--check-only` against the real smoke artifact returned the
intended error `Synthetic smoke artifacts must never access official test data`,
and created no test-evaluation directory.

Full-suite collection is blocked by missing `torch_pruning` and `plotly`; these
dependencies were not fabricated. Any available-subset run must state the ignored
modules and any additional failures separately. A read-only rerun on the unchanged
base commit reproduced eight pre-existing failures (six historical configuration
expectations and two Python 3.9 `zip(strict=True)` incompatibilities), with 69 tests
passing in those two files. Those unrelated historical files were not rewritten.
No CIFAR10 training, CUDA/GPU
training, official test evaluation, full 150-epoch run, server-version validation,
or whole-plan resume was performed as part of these local technical checks.
