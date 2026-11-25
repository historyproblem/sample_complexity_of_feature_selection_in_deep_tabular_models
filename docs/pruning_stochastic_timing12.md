# Night queue:720 epochs of stochastic-ST search

Exactly12 fresh searches run sequentially,60epochs each. This launcher defaults
to search-only and rejects full-training/test stages. There is no recovery in
this queue. Each selected source checkpoint remains available for a later90-epoch
recovery within the original150-epoch per-model budget.

Cartesian factors: warmup10/15, initiallambda0.001/0.01, soft_drop0.005/0.01/0.015.
Hard_drop stays0.03. Cases are ordered by warmup, initiallambda, then soft.

| Cases | Warmup | Initial lambda | Automatic update factor |
| --- | ---: | ---: | ---: |
| 1-3 | 10 | 0.001 | 2.511886 |
| 4-6 | 10 | 0.01 | 1.995262 |
| 7-9 | 15 | 0.001 | 6.309573 |
| 10-12 | 15 | 0.01 | 3.981072 |

Within each row, soft is0.005/0.01/0.015 (0.5/1/1.5percentagepoints); hard is
3percentagepoints. Initiallambda is applied both to model.lambda_coef and the
controller alpha_init. Warmup holds this coefficient fixed while training proceeds.
The unchanged auto rule targetslambda10 by the end ofepoch20 if every update
increases it, so warmup/init also change the derived step. Updates multiply or
divide by the table's factor. The cosine LR still spans60epochs.

Only the user-authorized soft/hard, warmup and initiallambda vary. A named timing
ablation marker permits only initial0.001/0.01 and warmup10/15; legacy validation
still requires its original0.001/10 recipe. Adaptive control, alpha_min1e-8,
alpha_max80, internal-only Gumbel-hard training/noisebeta1/temperature1,
deterministic-hard evaluation, floor0.08, label smoothing0/entropy0, seed42,
optimizer/scheduler and selection31-60/preBN remain unchanged. No size-aware
controller or5Mselection quota is added; the resulting sizes are unknown.

## Server command

```sh
if git fetch origin &&
   git switch feature/pruning-ablations-search-recovery &&
   git pull --ff-only &&
   (test -x .venv/bin/python || python3 -m venv .venv) &&
   .venv/bin/python -m pip install --no-cache-dir -r requirements.txt &&
   .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' &&
   .venv/bin/python scripts/run_stochastic_st_timing12.py --stage inspect; then
  nohup .venv/bin/python -u scripts/run_stochastic_st_timing12.py --stage search \
    > pruning_stochastic_timing12_search.log 2>&1 < /dev/null &
  tail -f pruning_stochastic_timing12_search.log
fi
```

All12recipes and pinned reference inputs are checked before training. The shared
zero-epoch initializer/reference cache from full4 is reused; missing inputs can
be restored from the original archive without a baseline fit. No old-queue wait.
Existing experiment folders are left intact; bounded epoch checkpoint retention
stays enabled. Finished searches are reused on retry; partial searches are
retained and refused rather than silently restarted.

Output: `outputs/runs/pruning_stochastic_timing12_20261125`. `[timing]` lines print
initiallambda/warmup/resolvedstep/factor before each search. `[search-summary]`,
`search_summary.csv` and `search_summary.json` report physical parameters,
remaining internal channels, pre-BN search-validation, fallback and timing factors.
`--dry-run` confirms12searches/720epochs/zero recovery and test without training.
