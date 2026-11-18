# Size ablations: search first, recovery later

Branch: `feature/pruning-ablations-search-recovery`.
All commands run from the repository root using `.venv/bin/python`.
This launcher does not start training a dense reference or access official test data.

Install dependencies using the existing standard file before launching:

```sh
.venv/bin/python -m pip install -r requirements.txt
```

For this Linux/CUDA branch, requirements.txt pins the working pair
torch2.6.0+cu126/torchvision0.21.0+cu126 and includes the official CUDA126 wheel
index. setup.py uses the matching2.6.0/0.21.0 base-version requirements, which
accept these CUDA builds. No extra installer or launcher flag is required.

## Matrix

One seed42 and fixed train/validation split42. Five soft/hard accuracy-drop pairs,
expressed as fractions, are in `configs/experiment/pruning_v3/ablations_grid.yaml`:

| Profile | Soft | Hard | Pilot |
| --- | ---: | ---: | --- |
| gap_005 | 0.00025 | 0.0005 | yes |
| gap_010 | 0.0005 | 0.001 | no |
| gap_050 | 0.0025 | 0.005 | yes |
| gap_150 | 0.0075 | 0.015 | no |
| gap_300 | 0.015 | 0.03 | yes |

Each profile has an independent internal search and an independent output search.
Internal search later feeds three recovery branches: inherited, scratch and random.
Output search later feeds one inherited recovery. Inherited is the shared control
for the three ablations, so four final models per profile are sufficient.

Random selection preserves the learned retained width of every gate group and
uniformly draws channel identities with a separately recorded deterministic mask
seed42. It inherits the surviving weights from the same search checkpoint.
This tests channel identities conditional on the learned width distribution.
It does not compare two complete architecture-search algorithms.

All recovery branches have fresh AdamW at1e-3/weight_decay5e-4 and fresh90-epoch
cosine to zero. Scratch resets every trainable parameter and BN state. Search uses
adaptive lambda, plain CE/label_smoothing0, beta0, floor0.08, gate threshold0.5,
warmup10 and an automatic log step (0.9210340371976183 for search60).

## Existing reference

The default reference directory is
`outputs/baseline/one_shot_dense_reference_seed42`. Each command also accepts
`--dense-source PATH_TO_EXISTING_REFERENCE` and `--output PATH`.
Use the same paths and subset for search, inspect and recover.
The reference must include the matching zero-epoch initializer and the complete
150-epoch dense validation history/state/config. No dense trained weights are
loaded into search, and no reference is trained implicitly.

On a server without an extracted reference, upload the existing archive to the
repository root and use this complete **first search command**:

```sh
.venv/bin/python scripts/launch_pruning_ablations.py --stage search --subset pilot \
  --reference-archive epoch_split_dense_reference_seed42_fresh_upload.tar.gz
```

It restores only the four required input files to the configured reference path,
then validates their compatibility before training. It refuses an existing
extraction destination; for an already extracted reference omit this option.
The local preflight verified this existing archive for both gate placements.
This reference is the fresh archive, not the earlier pinned prebn_repro4 reference;
all new branches share its matching zero-epoch initializer and validation curve.
After restoration, omit `--reference-archive` on subsequent commands.
If the reference is already extracted at
`outputs/runs/epoch_split_dense_reference_seed42_fresh`, use
`--dense-source outputs/runs/epoch_split_dense_reference_seed42_fresh` instead,
and keep that same option for inspect and recovery. No repeated upload is needed.

## First command: search only

Preview the six-search pilot without loading datasets or querying CUDA:

```sh
.venv/bin/python scripts/launch_pruning_ablations.py --stage search --subset pilot --dry-run
```

Run only its six60-epoch searches:

```sh
.venv/bin/python scripts/launch_pruning_ablations.py --stage search --subset pilot
```

Each search saves `selected_checkpoint.pt`, `selection.json`, resolved config,
controller/optimizer state, full search ledger and a physical export under
`outputs/runs/pruning_ablations_20261116/<profile>/<placement>/search/`.
There is no recovery training, extra BN calibration or official test inference.

Checkpoint selection uses only epochs31-60 for feasible and fallback selection.
It chooses minimum physical size among feasible candidates, or best validation
in the same window if none is feasible. Surviving channels retained by the floor
are opened when evaluating the actual physical predictor. The selected raw
checkpoint and all original channel indices remain frozen.

## Check search artifacts

```sh
.venv/bin/python scripts/launch_pruning_ablations.py --stage inspect --subset pilot
```

This prints selected epochs, validation accuracy, physical Params/MACs and fallback
status. It writes `search_summary_pilot.csv` and a checksum receipt
`search_review_pilot.json`. It checks every search before issuing the receipt,
including config identity, epoch window, selection/checkpoint identity, finished
60-epoch budget, pre-BN export and transfer-equivalence artifacts.

Use the validation summary to assess size coverage. No parameter quota or test
accuracy is used to obtain a desired size. Internal-only physical sizes can reach
a structural floor of3,683,893 parameters; output-only is limited to18,878,651.
Five tolerance settings need not produce five distinct sizes.

## Second training command: recovery only

After reviewing the pilot search results:

```sh
.venv/bin/python scripts/launch_pruning_ablations.py --stage recover --subset pilot
```

The launcher verifies all six searches against the inspection receipt **before
any recovery starts**. Changed inputs, configs or search files stop execution.
It reuses search rather than training it again; each physical branch receives
train-only BN200 followed by90training epochs. There are no gates or live sparsity
penalty during physical recovery; the search controller state is held in metadata.
Fresh optimizer state prevents E1 from conflating weight and optimizer inheritance.
Random export transfer and exact Params/MACs equality are checked before training.

The pilot has12final models:360search epochs in the first command,1080physical
training epochs in the second. Each model is charged60+90=150epochs even when its
selected checkpoint precedes epoch60. Search and recovery costs are recorded
separately, including actual updates/examples and reused search epochs.

## All five profiles

To train all ten searches before any recovery, use `--subset full` in all three
commands instead of `--subset pilot`. This yields20final models:600search epochs
then1800recovery epochs. Existing compatible completed pilot searches/recoveries
are checked and reused, never rerun. Profiles already searched must remain unchanged.
The two additional profiles can be adjusted using pilot validation/size evidence
before their first search; keep the same tolerance grid for internal/output.

An existing incomplete/failed run is retained and refused; there is no silent
restart or overwritten partial training. A completed cell is reused only after
artifact/config checks. Recovery is not automatically resumed in the middle of
an interrupted90-epoch branch.

No command above evaluates official test. Freeze the complete deployment list
before the separate final test/report step; validation values are not final test
quality. Historical test-informed comparisons remain exploratory.

## Two additional internal searches: quality thresholds only, 17 November

The current pair changes only the validation-quality tolerances driving lambda:

| Profile | soft_drop | hard_drop |
| --- | ---: | ---: |
| internal_gap001 | 0.00005 | 0.0001 |
| internal_gap0025 | 0.000125 | 0.00025 |

Both pairs are stricter than the completed internal/gap_005 control
(0.00025/0.0005). Lambda remains adaptive with its original initialization0.001,
minimum1e-8 and maximum80. The earlier cap0.1/1.0 proposal is withdrawn.
Profiles reject every training override except soft_drop/hard_drop. Resolved
configs are checked against internal/gap_005 with only these two fields removed.

Internal-only gates, seed42, existing reference/zero-epoch initializer,floor0.08,
threshold0.5,LS0,beta0,search60,pre-BN selection31-60,warmup10 and automatic log
step0.9210340371976183 are unchanged. Larger resulting sizes remain a hypothesis
until the searches finish; no parameter quota enters selection.

```sh
git pull --ff-only
.venv/bin/python scripts/launch_pruning_ablations.py \
  --config configs/experiment/pruning_v3/ablations_internal_less_sparse_pair.yaml \
  --stage search --subset full
```

This command trains exactly two new internal searches /120epochs, under
`outputs/runs/pruning_internal_threshold_pair_20261117/{internal_gap001,internal_gap0025}/internal/search/`.
The new directory separates the corrected threshold experiment from any previous
cap-based attempt. No existing outputs are overwritten or reused across the two
protocols. The same --config is required for later inspect/recovery.

The already restored reference is reused without --reference-archive. Recovery,
official test and baseline training are not started. Each eventual deployment
still has60search+90recovery=150attributed epochs. A process already launched with
the withdrawn cap config retains that old configuration until it is stopped;
a Git update cannot change a running Python process.

## Two intermediate-size probes, 18 November

The next internal-only pair aims to fill the6-7M and12-13M parameter ranges:

| Profile | Planning goal | soft_drop | hard_drop |
| --- | --- | ---: | ---: |
| internal_probe_6_7m | 6-7M | 0.000225 | 0.00045 |
| internal_probe_12_13m | 12-13M | 0.0001 | 0.0002 |

Only the two quality thresholds change from the existing internal recipe.
Lambda initialization/bounds,automatic step,seed42,reference/initializer,
floor0.08,gate threshold0.5,LS0,beta0,search60 and selection31-60/preBN0 are fixed.
The profile names are planning labels, not parameter quotas or selection criteria.
Observed size is not monotonic in the previous thresholds, so these two ranges
are exploratory goals and are not guaranteed before training.

```sh
git pull --ff-only
.venv/bin/python -u scripts/launch_pruning_ablations.py \
  --config configs/experiment/pruning_v3/ablations_internal_midrange_pair.yaml \
  --stage search --subset full
```

This starts exactly two new internal search-only runs /120epochs and reuses the
existing reference. New outputs are under
`outputs/runs/pruning_internal_midrange_pair_20261118/`; previous results are
preserved. For inspection use the same --config with --stage inspect. Recovery
remains a separate step; each eventual deployment stays within60+90=150epochs.
