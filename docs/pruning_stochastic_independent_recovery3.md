# Three inherited recoveries from the independent hard-threshold searches

Exactly three existing selected search checkpoints receive 90 recovery epochs
each. New training: 270 epochs. Each final model consumes its already completed
60 search + 90 recovery =150 epochs, then frozen official CIFAR-10 test. No new
search is permitted. The 20.306M hard=.02 fallback source is excluded.

| Source ID | Physical parameters | Retained internal channels | Search validation |
| --- | ---: | ---: | ---: |
| soft0025_hard050pp | 17,076,817 | 6375/7552 | 91.90% |
| soft0025_hard100pp | 11,282,074 | 4810/7552 | 91.06% |
| soft0025_hard300pp | 12,566,996 | 5237/7552 | 89.82% |

These measurements come from the user's search-summary screenshot. Before any
training, the launcher verifies all three original source folders, exact recipe,
reference/initializer/split hashes, selected checkpoint raw tensor hashes, masks,
physical parameter counts, retained-channel counts and non-fallback selection.
Missing/changed sources block the entire queue; no search or substitute baseline
is started. Source files remain under
`outputs/runs/pruning_stochastic_independent4_20261123/<id>/internal/search`.

Recovery inherits the selected surviving weights/mask, holds the saved adaptive
controller state, performs train-only BN200, and uses fresh AdamW and cosine90.
Physical recovery has no gates. Original search thresholds/selection policy,
floor.08, label smoothing0, entropy0 and all other training settings stay unchanged.
Bounded checkpoint storage stays on. Only the inherited branch runs; test follows
each frozen validation-selected deployment before the next recovery begins.

## Server command

```sh
if git fetch origin &&
   git switch feature/pruning-ablations-search-recovery &&
   git pull --ff-only &&
   (test -x .venv/bin/python || python3 -m venv .venv) &&
   .venv/bin/python -m pip install --no-cache-dir -r requirements.txt &&
   .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' &&
   .venv/bin/python scripts/run_stochastic_st_followup.py \
     --config configs/experiment/pruning_v3/stochastic_st_independent_recovery3.yaml --stage inspect; then
  nohup .venv/bin/python -u scripts/run_stochastic_st_followup.py \
    --config configs/experiment/pruning_v3/stochastic_st_independent_recovery3.yaml --stage run \
    > pruning_stochastic_independent_recovery3.log 2>&1 < /dev/null &
  tail -f pruning_stochastic_independent_recovery3.log
fi
```

The recovery-only protocol never waits for an old search process. Preflight reads
existing checkpoints without training. The same reference cache used by these
searches is required; no dense-reference fit is allowed.

Output: `outputs/runs/pruning_stochastic_independent_recovery3_20261124`.
Deployments: `<id>/internal/recovery/inherited/deployment.pt`.
Official test results accumulate in `test_evaluation/test_comparison.csv`.
`--dry-run` previews three recoveries/270 epochs/zero searches. `--stage test`
retries only missing frozen test evaluations after all fits complete. Completed
recoveries are reused; partial fits are retained and refused rather than restarted.
Comparisons remain exploratory after previous test results informed this series.
