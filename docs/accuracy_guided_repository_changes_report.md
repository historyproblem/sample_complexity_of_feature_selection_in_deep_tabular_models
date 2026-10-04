# Отчёт: «Часть 1. Изменение репозитория»

Ветка: `feature/accuracy-guided-gates-v3-20260914`. Основа —
`fix/resnet50-pruning-audit-20260905`, commit
`7e5cb63b938b605ef5a8bbb1db042452891c2a36`, указанный в приложенном ТЗ.
Ветка `audit-20260906` не обнаружена локально или среди удалённых веток origin;
эта неоднозначность сообщена пользователю до редактирования. Использована
найденная основа ТЗ. Работа выполнена в отдельном worktree; основной checkout,
его пользовательские изменения и исторические worktree не переключались.
Исходный и итоговый `git diff --binary` основного checkout совпали побайтово.

## Изменения по файлам

Пути указаны от корня репозитория.

| Файл | Изменённый контракт |
|---|---|
| `src/net_complexity/models/feature_selection.py` | RAW/EFFECTIVE/hard-decision API; immutable `n_b0`/`M0`; safe legacy width migration; канонический loss в фактическом posterior-path wrapper; диагностика коэффициентов. |
| `src/net_complexity/metrics/_channel_prob.py` | Survivor-метрики с весами по числу каналов; historical zero mass и macro mean отдельно; пустые множества — null/status. |
| `src/net_complexity/models/pruning_budget.py` | `select_learned_closed`: только raw-closed AND effective-closed survivors; original ids, floors/dependencies, подробные blocked/no-op отчёты; без квот. |
| `src/net_complexity/models/channel_pruning.py` | Передача optional `base_width` в существующий physical builder для tiny smoke. |
| `src/net_complexity/models/resnet.py` | Optional `base_width`, default 64 сохранён; tiny synthetic использует те же настоящие Bottleneck-блоки. |
| `src/net_complexity/models/pruned_bottleneck.py` | Такой же параметр ширины для реального sliced Conv/BN exporter; production defaults сохранены. |
| `src/net_complexity/training/adaptive_lambda.py` | Accuracy-only controller, paired gaps, три clock, cadence, bounds, идемпотентный rebase без сброса alpha. Legacy controller сохранён. |
| `src/net_complexity/training/engine.py` | Общий engine: фактический update/example ledger, атомарные epoch-events, used/next runtime, held recovery state, first-forward handoff hook, same-stage resume. |
| `src/net_complexity/training/run_history.py` | Атомарные полные v3 events, сохранение selection history; совместимый `zip` для локального Python 3.9. |
| `src/net_complexity/training/pruning_resume.py` | Fail-closed runtime/eval/same-stage resume, optimizer/scheduler/RNG/counters, явная legacy migration; неполные snapshots не выдаются за exact. |
| `src/net_complexity/training/pruning_measurement.py` | Изолированные RNG диагностики, physical/all-open/carry сравнение, отдельная gated export equivalence, фактические overhead examples/time калибровки. |
| `src/net_complexity/training/accuracy_guided_config.py` | Строгая отдельная v3 schema, проверка 150-epoch плана, конфликтов и reference/init provenance, read-only dry-run. |
| `src/net_complexity/training/accuracy_guided_pruning.py` | Итеративная orchestration через общий engine; best-feasible selection, recovery-first absolute guard, rollback/fallback, budget ledger, carry/rebase и provenance. |
| `src/net_complexity/training/pruning_synthetic.py` | Детерминированные synthetic данные, zero-epoch fixtures и явно искусственная reference-кривая; official-test sentinel. |
| `configs/accuracy_guided_gates_v3.yaml` | Единственный полный opt-in technical profile: accuracy_only, initial_channels, learned_closed_gates, carry, quality guard, 150 эпох. |
| `configs/accuracy_guided_gates_v3_smoke.yaml` | Отдельный короткий технический бюджет: S3/P/R1/S3/P/R1 = 8 эпох. |
| `scripts/launch_accuracy_guided_pruning.py` | Один запуск/конфиг и `--dry-run`; без job matrix, автоматического dense training и test evaluation. |
| `scripts/smoke_accuracy_guided_pruning.py` | Настоящий CPU optimizer/export/recovery smoke с двумя допустимыми удалениями и последующим no-op. |
| `scripts/evaluate_pruning_test.py` | Отдельный frozen v3 physical evaluator и `--check-only`; отказ для synthetic/gated-only artifacts. |
| `tests/test_accuracy_guided_gate_contract.py` | Loss/gradient identities, actual wrapper loss, normalization migration, metrics, selector, реальные Bottleneck transfer/cost проверки. |
| `tests/test_accuracy_only_runtime.py` | Controller cadence/rebase, used-next legacy bias, actual engine ledger и exact same-stage resume, interruption. |
| `tests/test_accuracy_guided_config.py` | Полный/smoke stage plan, strict conflicts, inputs, dry-run без CUDA/data/training. |
| `tests/test_accuracy_guided_iterative.py` | No-op, learned pruning, carry/rebase, recovery/calibration, quality rejection, rollback/fallback, technical failure, partial interruption. |
| `tests/test_accuracy_guided_evaluator.py` | Frozen physical artifact validation, типы/хеши/width metadata; запрет test доступа synthetic selection. |
| `docs/accuracy_guided_gates_v3.md` | Контракт, миграция, ограничения, provenance и команды проверки. |
| `docs/accuracy_guided_repository_changes_report.md` | Этот отчёт и фактически выполненные проверки. |

Исторические v1/v2/param-budget конфиги, launchers, результаты и предупреждения
`AGENTS.md` не переписаны и не переименованы.

## Фактическая формула и runtime

```text
L_gate = alpha / M0 * sum_b sum_j(m_bj * p_raw_bj) / n_b0
lambda_effective_b = alpha * n_bt / n_b0
```

`alpha` — один скаляр accuracy-controller. Второго умножения alpha на survivor
ratio нет. Для n0=100, n=50, alpha=0.001 effective lambda=0.0005; коэффициент
на survivor остаётся 0.00001 до неизменного множителя `1/M0`.
Формула проверена на настоящем `ClassificationFeatureSelectionWrapper.forward`
с entropy coefficient=0, а не только в YAML или отдельной формуле.

Полный текущий план сохранён:
`S20 → P → R15 → S20 → P → R15 → S20 → P → R60 = 150`.
P может удалить ноль каналов. Все p=0.9 дают no-op; два допустимых closed gates
дают только два удаления. Recovery не содержит gates/penalty, но сохраняет
выбранную alpha для следующего search. Stage optimizer/scheduler перезапускаются;
сохранение Conv/BN не называется сохранением optimizer state.

Quality guard использует `A_ref(end_global_epoch)-hard_drop` после выделенного
recovery. При rollback rejected recovery остаётся в consumed ledger. Сохранённое
происхождение deployment указывает на реально accepted checkpoint даже при
отклонении последней стадии. Контроллер стремится удерживать качество, но не
гарантирует соблюдение допуска.

## Реально выполненные проверки

- Локальная среда: Python 3.9.6, torch 2.8.0, pytest 8.4.2, CPU.
- Полный `pytest -q`: collection остановлен отсутствующими `torch_pruning` и
  `plotly`. Зависимости не подменялись фиктивными импортами.
- Повторный suite с явным исключением только `test_depgraph_pruning.py` и
  `test_last_experiment_plotting.py`: **634 passed, 8 failed**, 95.17 s.
- Все восемь оставшихся failures независимо воспроизведены на исходном
  `7e5cb63`: шесть ожиданий исторических конфигов в `test_experiment_configs.py`
  и два `zip(strict=True)` в `test_gradient_norm_logging.py` под Python 3.9.
  На исходной ветке эти два файла дали **69 passed, 8 failed**.
- Пять новых test files проверены отдельным финальным запуском:
  **92 passed**, 26.03 s. Включена дополнительная проверка, что фактический
  `reg_loss` при alpha=1e-8 не теряется из-за вычитания близких FP32 total loss/CE.
- Standalone CPU smoke выполнен: 8 training epochs, 6 search epochs,
  16 optimizer updates, 64 training examples, 2 materialized channels,
  1 carry/rebase, второй commit — no-op. Synthetic reference имеет ноль
  реальных dense-training epochs и не является результатом качества.
- Base dry-run: 150 epochs, training=false, CUDA=false, evaluate_test=false;
  четыре отсутствующих reference/init файла явно помечены blocked.
- Frozen evaluator `--check-only` на фактическом smoke artifact вернул ожидаемый
  отказ `Synthetic smoke artifacts must never access official test data`;
  official test и создание test-report не выполнялись.
- `git diff --check` и компиляция новых runtime/smoke модулей прошли. Для
  компиляции использован writable `PYTHONPYCACHEPREFIX`: системный Python cache
  вне sandbox недоступен.

Постоянный worktree расположен в `outputs/pro_model_review/accuracy_guided_gates_v3`
основного checkout. Внутри него реальные логи, dry-run JSON и smoke artifacts
сохранены в `outputs/verification/20260915/` (артефакты не добавляются в Git).

## Команды проверки

Из корня checkout новой ветки, на сервере через `.venv/bin/python`:

```sh
.venv/bin/python scripts/launch_accuracy_guided_pruning.py --dry-run
.venv/bin/python -m pytest -q tests/test_accuracy_guided_gate_contract.py tests/test_accuracy_only_runtime.py tests/test_accuracy_guided_config.py tests/test_accuracy_guided_iterative.py tests/test_accuracy_guided_evaluator.py
.venv/bin/python scripts/smoke_accuracy_guided_pruning.py --output /tmp/accuracy-guided-v3-smoke
.venv/bin/python scripts/evaluate_pruning_test.py --protocol-v3 --run-dir PATH_TO_COMPLETED_V3_RUN --check-only
```

Полная доступная часть suite с честно указанными исключениями:

```sh
.venv/bin/python -m pytest -q --ignore=tests/test_depgraph_pruning.py --ignore=tests/test_last_experiment_plotting.py
```

Отдельный official-test evaluator для выбранного deployment после будущего
полного запуска (команда приведена, сейчас не выполнялась):

```sh
.venv/bin/python scripts/evaluate_pruning_test.py --protocol-v3 --run-dir PATH_TO_COMPLETED_V3_RUN --data data --device cpu --output PATH_TO_FRESH_TEST_REPORT
```

## Ограничения и оставшаяся интеграция

Exact resume реализован для завершённой эпохи той же стадии. Восстанавливаются
optimizer/scheduler/RNG, runtime, mask, controller, ledger и предыдущие кандидаты
selection. Partial-epoch и persistent-worker snapshots отвергаются явно.
**Whole-plan transaction resume не реализован**; `--resume-from` не начинает
запуск заново под видом продолжения.

Полный профиль blocked до предоставления совместимых immutable reference и
zero-epoch initializer. Проверка полного 150-epoch CIFAR10/GPU runtime, official
test и проверка в заявленной зависимости Python>=3.10/torch2.10 не выполнялись.
CPU smoke не служит доказательством ResNet50/CIFAR accuracy или GPU memory/latency.
Backward FLOPs не измерялись; forward MAC, actual counters, process RSS и overhead
имеют явно указанный scope.

Часть 2 намеренно не реализована: нет новых baseline, reset/reopen gates,
DepGraph-сравнений, O30/O60/O90/OC/N0, I_reset/I_reopen, seed-grid, sweep,
confirmation manifest или матрицы/порядка исследовательских запусков.
