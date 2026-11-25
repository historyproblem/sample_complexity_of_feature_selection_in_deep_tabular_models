# Eight inherited recoveries from the completed timing12 study

Exactly eight90-epoch recoveries,720new training epochs, with frozen official test
immediately after each model. No new search, no scratch/random branches and no
old-process wait. Every final model stays within60search+90recovery=150epochs.

Selection uses search size/validation only: all six non-fallback sources plus the
two compact fallback sources nearest the88.94% search threshold. These miss by
only0.26/0.24percentagepoints. The four larger/weaker warmup15 fallback sources
are excluded. The two approximately5M models run first.

| Order | Source ID | Parameters | Channels retained | Search validation | Fallback |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | w10_l010_s010 | 5,110,943 | 1688 | 90.70% | false |
| 2 | w10_l010_s015 | 5,174,199 | 1700 | 89.08% | false |
| 3 | w10_l010_s005 | 6,415,681 | 2477 | 89.40% | false |
| 4 | w10_l001_s015 | 7,045,869 | 2869 | 88.94% | false |
| 5 | w10_l001_s010 | 7,398,628 | 3037 | 88.68% | true |
| 6 | w10_l001_s005 | 7,983,520 | 3292 | 88.70% | true |
| 7 | w15_l010_s005 | 9,680,738 | 4172 | 89.24% | false |
| 8 | w15_l001_s005 | 11,883,222 | 5018 | 89.04% | false |

Before any training, all eight source folders/checkpoints are checked, including
raw tensor hashes, config/reference/initializer/split hashes, mask identity and
measured parameter/channel/selected-epoch/validation/fallback pins from the supplied
CSV. Sources stay at `outputs/runs/pruning_stochastic_timing12_20261125/<id>/internal/search`.
A missing or changed source blocks the whole queue; no search is restarted.

Each recovery inherits its original selected mask and surviving weights. Original
warmup, initiallambda and thresholds are retained as provenance, not rerun. The
structural model has no gates; its adaptive state is held, with train-only BN200,
fresh AdamW and cosine90. Label smoothing0/entropy0/floor.08 and the rest of the
recipe remain unchanged. Bounded checkpoint retention stays enabled. Final
checkpoint selection uses validation; official test performs no training/BN updates.

## Server command

```sh
if git fetch origin &&
   git switch feature/pruning-ablations-search-recovery &&
   git pull --ff-only &&
   (test -x .venv/bin/python || python3 -m venv .venv) &&
   .venv/bin/python -m pip install --no-cache-dir -r requirements.txt &&
   .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' &&
   .venv/bin/python scripts/run_stochastic_st_followup.py \
     --config configs/experiment/pruning_v3/stochastic_st_timing_recovery8.yaml --stage inspect; then
  nohup .venv/bin/python -u scripts/run_stochastic_st_followup.py \
    --config configs/experiment/pruning_v3/stochastic_st_timing_recovery8.yaml --stage run \
    > pruning_stochastic_timing_recovery8.log 2>&1 < /dev/null &
  tail -f pruning_stochastic_timing_recovery8.log
fi
```

Output: `outputs/runs/pruning_stochastic_timing_recovery8_20261125`.
Deployments: `<id>/internal/recovery/inherited/deployment.pt`.
Test results: `test_evaluation/test_comparison.csv`. `--dry-run` confirms8recoveries/
720epochs/0search. `--stage test` retries missing test inference after all fits
complete. Completed fits are reused; partial fits are retained and refused rather
than silently restarted. Comparisons remain exploratory after earlier tests.
