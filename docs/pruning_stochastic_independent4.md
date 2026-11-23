# Four searches with independent soft/hard tolerances

Hold soft_drop=0.00025 (0.025 percentage points) fixed, matching the previous
9.053M profile. Change only hard_drop; no longer keep hard=2*soft.

| ID | soft_drop | hard_drop | Hard tolerance, percentage points |
| --- | ---: | ---: | ---: |
| soft0025_hard050pp | 0.00025 | 0.005 | 0.5 |
| soft0025_hard100pp | 0.00025 | 0.01 | 1.0 |
| soft0025_hard200pp | 0.00025 | 0.02 | 2.0 |
| soft0025_hard300pp | 0.00025 | 0.03 | 3.0 |

Larger hard tolerances delay lambda rollback and loosen candidate validation
admissibility. They do not guarantee fewer channels or preserved final accuracy.
The experiment is a one-factor hard-threshold sweep with a fixed soft threshold.

## First stage: exactly four 60-epoch searches

Run with `--stage search`: total240 new epochs. No recovery, official test,
scratch/random branches or old-process waiting. Each search starts from the same
pinned zero-epoch initializer. Selected source checkpoints remain available for
later90-epoch inherited recovery, within150 total epochs per final model.

The resolved training recipe differs from the prior9.053M profile only in
hard_drop. Internal gates, Gumbel-hard ST search training (noise beta1/temperature1),
deterministic-hard validation, adaptive lambda init.001/min1e-8/max80, floor.08,
label smoothing0/entropy0, seed42, optimizer/scheduler, selection epochs31-60 and
physical pre-BN scoring stay unchanged. Bounded checkpoint retention stays on.
Reference inputs reuse the existing full4 `reference_inputs` cache with unchanged
content hashes; no dense baseline is trained and no learned full4 weights reused.

```sh
if git fetch origin &&
   git switch feature/pruning-ablations-search-recovery &&
   git pull --ff-only &&
   (test -x .venv/bin/python || python3 -m venv .venv) &&
   .venv/bin/python -m pip install --no-cache-dir -r requirements.txt &&
   .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' &&
   .venv/bin/python scripts/run_stochastic_st_full4.py \
     --config configs/experiment/pruning_v3/stochastic_st_independent4.yaml --stage inspect; then
  nohup .venv/bin/python -u scripts/run_stochastic_st_full4.py \
    --config configs/experiment/pruning_v3/stochastic_st_independent4.yaml --stage search \
    > pruning_stochastic_independent4_search.log 2>&1 < /dev/null &
  tail -f pruning_stochastic_independent4_search.log
fi
```

Output: `outputs/runs/pruning_stochastic_independent4_20261123`. After each search,
the console prints `[search-summary]` with physical parameters, retained internal
channels, physical pre-BN search-validation and fallback status. The final table
also appears in the console. `search_summary.csv`, `search_summary.json` and
`search_state.json` are saved at the output root. Missing sparse mask groups count
as fully retained, so the common denominator is7552. These are search-validation
results, not final test metrics.

Use `--stage search --dry-run` to preview4searches/240epochs/0recovery/0test without
training. Retrying the identical command reuses finished searches; partial runs
are retained and refused, never silently restarted. The selected source `.pt`
files and candidate JSONL records are preserved. Inspect channels before choosing
which models receive the remaining90 recovery epochs. Omitting `--stage search`
uses the historical full-training mode, so keep the explicit stage in this command.
