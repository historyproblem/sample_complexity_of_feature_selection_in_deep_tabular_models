# Selected MobileNet recovery and four fresh mask searches

Run from the repository root through `.venv/bin/python`:

```bash
nohup .venv/bin/python -u scripts/run_mobilenet_recovery_wide_grid.py \
  --baseline-archive outputs/baselines/mobilenet_baseline_partial.tar.gz \
  > outputs/logs/mobilenet_recovery45_wide_search.log 2>&1 < /dev/null &
```

The runner first physically removes the selected 962 channels from the
threshold0.9/soft0.06/hard0.10 epoch30 architecture (2,251,882 parameters),
copies surviving learned weights and BatchNorm statistics from its existing
`checkpoints/best.pt`, and fine-tunes for45epochs. AdamW and cosine45 restart;
backbone LR is0.0001 and classifier LR0.001. The original controller state is
preserved in `source_provenance.json`; this structural model has no gates.
Label smoothing remains0. Validation selects the checkpoint for frozen test.

Then four independent30epoch search-only runs start from ImageNetV2, fresh
masks, lambda0.0001 and a new adaptive controller: thresholds0.5/0.9 crossed
with soft/hard accuracy drops0.15/0.20 and0.20/0.25. Warmup10, automatic fixed
log step0.575646273249, no step boost, no parameter cap, channel pruning,
seed42 and label smoothing0 are retained. No45epoch phase follows these four.

This adds165epochs across five models. The recovered model's complete
history is30+45=75epochs; each new search uses30. New searches do not inherit
the recovered weights, masks or controller. The historical reference is
reused without training a new baseline.

The original server search directory, including its checkpoint, must remain
available under `outputs/runs/mobilenet_onepass_grid_thr05_09_search_only_warmup10/`.
Metadata-only downloaded archives do not contain those weights. Missing or
different source artifacts fail before training; there is no weights fallback.
`--source-profile` can point to a relocated complete copy of that profile.

New outputs are under
`outputs/runs/mobilenet_recovery45_then_wide_search_20261112/`.
Completed matching stages are reused; incomplete or changed stages refuse
automatic repetition. `--dry-run` prints commands without CUDA or training.
