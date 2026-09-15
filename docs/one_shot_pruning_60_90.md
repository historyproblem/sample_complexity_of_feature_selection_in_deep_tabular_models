# One shared 60-epoch search: inherited versus scratch compact models

This is the single separately authorized `pruning_v3_one_shot_60_90` comparison.
It adds its own runner and strict configuration adapter; the existing iterative
accuracy-guided protocol and historical experiments are unchanged. No other
schedule, baseline, DepGraph comparison, reset/reopen gate policy, or parameter
grid is introduced.

The full profile is
`configs/experiment/pruning_v3/one_shot_60_90_inherited_vs_scratch.yaml`. It performs
one adaptive gated search for 60 epochs from the verified zero-trained-epoch
initializer, selects one validation-feasible checkpoint and its learned mask,
and builds two physically identical compact models:

| Branch | Initial physical weights and BN | Physical training |
|---|---|---:|
| `inherited` | Surviving Conv/BN tensors from the selected search checkpoint | 90 epochs |
| `scratch` | Fresh PyTorch default initialization of all trainable parameters and BN state | 90 epochs |

Both branches use the **same selected checkpoint identity and original-coordinate
mask** to define their architecture. Scratch does not retain inherited Conv,
classifier, BN affine weights, running statistics, or batch counters. Removing
gates means both physical branches have zero gate penalty and ordinary weight
decay; the shared search retains mandatory accuracy-only adaptive lambda.
Each branch starts a new AdamW optimizer and a 90-epoch cosine schedule. There is
no second search, hidden warmup, gate reentry, BN calibration, or extra recovery.

The attribution is 60 shared search epochs + 90 physical epochs = **150 epochs
per branch**. Actual work performed by the combined run is 60 + 90 + 90 = **240
unique training epochs**. The shared search executes once, and its consumed
updates/examples are charged explicitly when comparing either branch. Selecting
an early checkpoint does not rewind work already consumed. Equal epochs are not
reported as equal training FLOPs.

## Shared selection and export-only diagnostics

Search uses the existing v3 `accuracy_only` controller, initial-width loss,
learned-closed-gate eligibility and `best_feasible_compact` selection. All search
checkpoints use the same dense validation reference at epoch 60. Neither the mask
nor the checkpoint may be independently reselected for scratch. If no feasible
search checkpoint exists, the run records `no_feasible_search` and stops before
export/branch training; it does not launch an alternative experiment.

Before either branch trains, export-only diagnostics compare the selected gated
predictor with the physical inherited export. A separate all-open transfer check
uses the retained topology to distinguish tensor-transfer errors from the opening
jump caused by any blocked closed survivors.
These are frozen validation passes with no gradients, optimizer steps, BN updates,
or calibration. Scratch initialization and diagnostics are distinguished from
inherited export: their different functions are expected, while their physical
topology and cost must agree. Technical transfer failures abort the run rather
than being hidden by training. An unexplained gated/physical mismatch also aborts
before branch training and retains `export_only/diagnostics.json`. A measured
mismatch caused by blocked closed survivors is recorded explicitly; it does not
change the common architecture or trigger an alternative experiment.

Each physical branch selects its frozen checkpoint using validation. Final quality
is reported against the immutable dense reference at the completed branch budget.
An infeasible branch retains the common selected architecture and is labelled
infeasible; it does not invoke the iterative runner's rollback/fallback policy.
The inherited `accuracy_guided.guard` keys support the shared base-schema check;
the one-shot runner's operative policy is stated explicitly in dry-run
`execution_policy`, including `iterative_recovery_guard_executed=false`.

The base seed/split 42, ResNet50 internal-channel gates, threshold 0.5,
min_keep_ratio 0.5, CIFAR10 45,000/5,000 training/validation split, batch 128,
augmentations and AdamW settings remain inherited from the corrected v3 profile.
The reference and initializer paths are also unchanged. Missing reference/input
files block execution; a dense reference is never trained automatically.

## Commands and output paths

All commands run from the repository root through its virtual environment.
Preview the exact resolved protocol, budget attribution, inputs and output paths:

```sh
.venv/bin/python scripts/launch_one_shot_pruning.py --config-name experiment/pruning_v3/one_shot_60_90_inherited_vs_scratch --output outputs/runs/one_shot_60_90_inherited_vs_scratch --dry-run
```

The full launch command is provided for execution after technical review. It is
not part of the local implementation verification:

```sh
.venv/bin/python scripts/launch_one_shot_pruning.py --config-name experiment/pruning_v3/one_shot_60_90_inherited_vs_scratch --output outputs/runs/one_shot_60_90_inherited_vs_scratch
```

The explicit `--config-name` and `--output` shown above are also the launcher
defaults. An existing output directory is refused, including an empty directory;
there is no silent overwrite or implicit resume. Relocated reference/initializer
paths can be supplied together with `--dense-source`, subject to the same strict
compatibility checks. The timestamped default paths refer to a historical server
run; Git does not distribute those training artifacts. Locate the existing files
from the repository root if their directory is unknown:

```sh
find outputs -type f \( -name shared_random_seed42.pt -o -name pilot_state.json \) -print
```

Pass the directory containing both `J1_dense_control/` and
`shared_random_seed42.pt`, or the `J1_dense_control/` directory itself. Replace
the example absolute path below with the actual location:

```sh
.venv/bin/python scripts/launch_one_shot_pruning.py --dense-source /absolute/path/to/existing_dense_run --output outputs/runs/one_shot_60_90_inherited_vs_scratch --dry-run
```

After this reports `inputs.status=ready`, use the same command without `--dry-run`
to launch. The expected metadata in `J1_dense_control/` are `global_history.csv`,
`pilot_state.json` and `resolved_config.yaml`. A self-contained historical adaptive
bundle is also supported: its root-level `adaptive_reference_history.csv` and
`J1_dense_control_resolved.yaml` replace the history/config pair when both original
files are absent. The state and original initializer retain their usual paths.
Incomplete metadata pairs are not mixed, and stale `reused_from` paths are not
followed automatically. A nightly directory containing only a reused dense state
is insufficient; restore the original artifacts from storage if they are missing.

Individual `--override KEY=VALUE` arguments take precedence over `--dense-source`
paths when files are stored separately. The initializer must still match the dense
state's `common_init_hash` and have `trained_epochs=0`; neither a trained dense
checkpoint nor a newly generated random file is a replacement. No dense deployment
weights are loaded during this preflight. Missing inputs produce an error listing
the absolute missing paths and stop before training or creating the output run.

The full schema rejects another search
length, extra branches, fixed lambda, pruning targets and enabled calibration.
Short synthetic plans require the existing explicit smoke marker and separate
fixture inputs; they do not create additional experimental profiles.

For the default root
`outputs/runs/one_shot_60_90_inherited_vs_scratch/`:

| Path | Purpose |
|---|---|
| `one_shot_state.json`, `resolved_config.yaml` | Overall protocol state, actual combined cost ledger and frozen configuration |
| `shared_search/training/` | The single gated search and its checkpoints/history |
| `selected_checkpoint.pt`, `selection.json` | One selected checkpoint identity and mask shared by both branches |
| `export_only/diagnostics.json`, `export_only/transfer_equivalence.json` | Frozen diagnostics before any physical-branch training |
| `export_only/deployment.pt` | Physical inherited export before its 90 training epochs |
| `inherited/training/`, `scratch/training/` | Independent physical branch checkpoints/history |
| `inherited/initial_state.pt`, `scratch/initial_state.pt` | Exact branch initial tensors/BN state and initialization hashes |
| `inherited/deployment.pt`, `scratch/deployment.pt` | Validation-selected frozen physical deployments |
| `inherited/branch_state.json`, `scratch/branch_state.json` | Branch provenance, quality and consumed ledger |
| `comparison.json` | Validation and feasibility summary; official test remains absent |
| `test_evaluation/` | Separate official-test reports for the explicit evaluator command below |

The independent evaluator validates both branch artifacts and their common
selection before constructing one official test loader. A check-only pass uses
no test data:

```sh
.venv/bin/python scripts/evaluate_one_shot_pruning_test.py --run-dir outputs/runs/one_shot_60_90_inherited_vs_scratch --check-only
.venv/bin/python scripts/evaluate_one_shot_pruning_test.py --run-dir outputs/runs/one_shot_60_90_inherited_vs_scratch --data data --device cpu --output outputs/runs/one_shot_60_90_inherited_vs_scratch/test_evaluation
```

Official test is never used for search, mask choice or branch checkpoint selection.
Frozen evaluation performs no training or BN updates and verifies unchanged
weights/buffers. Synthetic artifacts are refused. Validation accuracy is never
substituted for missing test accuracy, and comparisons informed by earlier test
results remain exploratory.
If `--output` is omitted from the evaluator command, its default is
`RUN_DIR/one_shot_test_evaluation`; the training launcher never creates either
test-report directory.

## Technical verification

The schema/CLI checks include exact full schedule and common topology constraints,
separation from the unchanged iterative validator, no calibration, no test access,
clear shared-work accounting, missing-input blocking and refusal of existing output.
Dry-run spies reject training-runtime imports, CIFAR construction and CUDA calls.

```sh
.venv/bin/python -m pytest -q tests/test_one_shot_pruning_config.py tests/test_one_shot_pruning.py tests/test_one_shot_pruning_evaluator.py
```

Local synthetic tests verify the actual compact runtime and export behavior; they
are not evidence about CIFAR10 quality. No full 60+90 experiment or official test
evaluation is run while implementing this change. The verification report must
state the actual environment/dependency versions and any unperformed checks.

The initial implementation's combined run passed **228 tests in 56.34 seconds**: the three one-shot
suites above, plus `test_accuracy_guided_config.py`,
`test_accuracy_guided_iterative.py`, `test_accuracy_guided_evaluator.py`,
`test_accuracy_guided_gate_contract.py`, `test_pruning_resume.py` and
`test_pruning_test_evaluation.py`. Eight one-shot runtime tests include actual
tiny CPU search/final optimizer steps, no-op and learned pruning, complete state
inheritance/reinitialization, consumed-budget accounting and both explained and
unexplained export mismatches.

The relocated-input preflight update passed **137 tests in 21.72 seconds** using
the three one-shot suites plus `test_accuracy_guided_config.py`. This includes 55
configuration/CLI tests, actual relocated zero-epoch fixtures, unchanged initializer
hash checks, a tripwire against loading trained dense checkpoints, and both known
reference layouts. A real CLI invocation with missing inputs exited with code 2,
listed all four absolute paths, and did not create the output run or start training.

The local verification environment was Python 3.9.6, PyTorch 2.8.0 and pytest 8.4.2
on CPU (CUDA unavailable). The full-profile dry-run completed without training
and reported four missing inputs: the dense reference CSV, state JSON, resolved
configuration and `shared_random_seed42.pt`. These must exist at the configured
server paths, or be supplied through the documented path overrides. No replacement
dense checkpoint or initializer was created for the full profile.
