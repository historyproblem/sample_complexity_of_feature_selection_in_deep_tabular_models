# Recovery for the first two quality17m3 checkpoints

The completed search log identifies these two non-fallback sources:

| ID | Parameters | Remaining channels | Selected search epoch | Search validation |
| --- | ---: | ---: | ---: | ---: |
| tighter_hard | 18,344,554 | 6566/7552 | 33 | 91.58% |
| earlier_pressure | 17,462,511 | 6462/7552 | 34 | 91.46% |

Only these two checkpoints receive90 inherited recovery epochs each, followed
by official test before the next recovery. Both original searches completed60
epochs; selected epochs33/34 do not reduce their already consumed search budget.
New training is180 epochs, and total training stays150 per model. The third
`combined` source is excluded; its presence or completion is not required.

The recovery-only protocol forbids a new search and does not wait for old queues.
Preflight verifies both original source folders before training: source recipe,
reference/initializer/split identities, checkpoint SHA256 prefixes from the log,
raw tensor identities, selected epochs, sizes, channels, validation and absence
of fallback. Keep the source folders in their original locations, including
selected_checkpoint.pt and export_only/deployment.pt. Missing or changed sources
block the whole queue without training a substitute.

The inherited physical recovery recipe is unchanged: fixed architecture/mask,
surviving weights, train-only BN200, fresh AdamW and cosine90, LS0, no gates or
gate penalty, held adaptive-controller state. Search thresholds remain exactly
those of each source:0.00025/0.004 and0.001/0.005. Selection remains validation-only
with the original pre-BN search window31-60. Bounded checkpoint retention stays on.

## Server command

Run from the repository root:

```sh
if git fetch origin &&
   git switch feature/pruning-ablations-search-recovery &&
   git pull --ff-only &&
   (test -x .venv/bin/python || python3 -m venv .venv) &&
   .venv/bin/python -m pip install --no-cache-dir -r requirements.txt &&
   .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))' &&
   .venv/bin/python scripts/run_stochastic_st_followup.py \
     --config configs/experiment/pruning_v3/stochastic_st_quality17m_recovery2.yaml --stage inspect; then
  nohup .venv/bin/python -u scripts/run_stochastic_st_followup.py \
    --config configs/experiment/pruning_v3/stochastic_st_quality17m_recovery2.yaml --stage run \
    > pruning_stochastic_quality17m_recovery2.log 2>&1 < /dev/null &
  tail -f pruning_stochastic_quality17m_recovery2.log
fi
```

Output: `outputs/runs/pruning_stochastic_quality17m_recovery2_20261127`.
Deployments: `<id>/internal/recovery/inherited/deployment.pt`.
Official test: `test_evaluation/test_comparison.csv`.
`--stage run --dry-run` previews two recoveries/180 epochs/zero searches.
`--stage test` retries missing frozen test evaluations after both fits finish.
Completed recoveries are reused; partial fits are retained and refused rather
than silently restarted. Earlier exploratory test results informed this study.
