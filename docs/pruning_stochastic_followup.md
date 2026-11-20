# Stochastic ST: three recoveries and two new models

Run from the repository root using `.venv/bin/python`. Configuration:
`configs/experiment/pruning_v3/stochastic_st_followup5.yaml`.

Exactly five main-method models are processed sequentially, with one inherited
branch each. There are no scratch or random-mask recovery ablations in this queue.

| Order | Model | New search epochs | Recovery epochs | soft_drop | hard_drop |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Almost-dense second search, 23,520,702 parameters | 0 | 90 | 0.0001 | 0.0002 |
| 2 | Pruned first search, 11,678,972 parameters | 0 | 90 | 0.000225 | 0.00045 |
| 3 | Third search, `internal_gap001`; measured size read from its export | 0 | 90 | 0.00005 | 0.0001 |
| 4 | New `gap_003_006pp` | 60 | 90 | 0.0003 | 0.0006 |
| 5 | New `gap_005_010pp` | 60 | 90 | 0.0005 | 0.001 |

The "full" control here is the existing almost-dense stochastic search with four
removed channels. Its mask is preserved; it is not replaced with an empty mask or
the separately trained 150-epoch dense reference. Source sizes/epochs/checksum
prefixes are measured identity checks, never parameter quotas. The third source
and two new model sizes are not guessed.

Total new training: `3 * 90 + 2 * (60 + 90) = 570` epochs. Each resulting deployment
uses its 60 consumed search epochs plus 90 recovery epochs, at most 150 total.

The new searches differ only in the two accuracy tolerances. All models preserve
internal-only gates, stochastic-ST `gumbel_hard` training, noise scale 1,
temperature 1, deterministic-hard validation, threshold 0.5, floor 0.08, entropy
coefficient 0, label smoothing 0, seed/split 42, and the existing optimizer,
scheduler and adaptive-lambda settings (initial 0.001, minimum 1e-8, maximum 80).
Both feasible and fallback selection retain epochs 31-60 and pre-BN scoring.
New sizes and quality are exploratory outcomes, not promises.

## Launch

```sh
git pull --ff-only
nohup .venv/bin/python -u scripts/run_stochastic_st_followup.py \
  > pruning_stochastic_followup5.log 2>&1 < /dev/null &
tail -f pruning_stochastic_followup5.log
```

The runner first waits for the existing `ablations_stochastic_st4.yaml --stage search`
process owned by the current user to exit. It neither stops that process nor runs
another GPU training alongside it. All three selected source folders must then
have finished exports, including `selected_checkpoint.pt`. Before the first new
optimizer step, every source config, selection, file checksum, raw tensor hash,
reference/initializer and split identity is checked. The first two sources are
also checked against the parameter counts, selected epochs and checksum prefixes
already present in the supplied log. Actual source sizes and physical pre-BN
validation scores are printed before training.

Preview without waiting, reading datasets or training:

```sh
.venv/bin/python scripts/run_stochastic_st_followup.py --dry-run
```

Use `--stage inspect` for a source-only preflight. `--dense-source PATH` can relocate
the same immutable reference; its hashes remain pinned. This workflow never fits
a new dense reference. `--skip-wait` is only for cases where the old queue has
already exited and process discovery is unavailable.

## Results and repeat commands

Output root: `outputs/runs/pruning_stochastic_followup5_20261121`.
Original search folders remain unchanged. Each model uses
`<id>/internal/recovery/inherited/deployment.pt`; new searches also retain
`<id>/internal/search/selected_checkpoint.pt` and their selected export.

After each model finishes recovery, its checkpoint selected by validation is
frozen and evaluated on all 10,000 official CIFAR-10 test examples before the next
fit begins. Inference uses no gradients, parameter updates, BatchNorm updates or
gates, and preserves subsequent training RNG states. Results, frozen manifests
and predictions accumulate in `test_evaluation/`; the log prints `[official-test]`.
The comparison is exploratory because earlier test results informed experiments.

`followup_state.json` records progress; `source_review.json` pins the three reused
sources. Completed matching fits are reused, partial fits are retained/refused
instead of silently consuming another 90 epochs. Finalized recovery epoch files
are cleaned by default; initial, best and deployment artifacts remain. Use
`--keep-epoch-checkpoints` to retain all epoch files.

If only test inference needs repeating after all five fits complete:

```sh
.venv/bin/python scripts/run_stochastic_st_followup.py --stage test
```

This uses completed recovery artifacts and verified cached reports. Original
reference/source search folders need not be present; no training is repeated.
