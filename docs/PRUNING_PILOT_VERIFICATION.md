# Проверки от 5 сентября 2026

## Исправление старта ResNet50 с channel history, 6 сентября

Серверный D1 остановился до первой эпохи: конфиги включали
`run_history.log_channel_history`, но пара ResNet50 / MaskedGumbelBottleneckLayer
не была зарегистрирована в channel-history collector. Прежний synthetic smoke
отключал эту настройку и поэтому не покрывал реальный путь запуска логирования.

- Зарегистрирован существующий совместимый Gumbel collector для этой пары.
  Он сохраняет выходные и mid1/mid2 каналы, включая исходные индексы и маски.
  На структурном recovery без gates остаётся только заголовок CSV.
- Все семь pilot-конфигов теперь проверяются через настоящий RunHistory
  с включённым channel history. До исправления все семь проверок воспроизвели
  серверный ValueError; после исправления прошли.
- Smoke больше не отключает channel history и проверяет содержимое CSV.gz
  после каждой стадии. CPU smoke обоих профилей прошёл: D1/D2 по две маленькие
  эпохи, J1/J2/J3 по четыре; test-sentinel сохранён. Это синтетические данные,
  не измерение качества на CIFAR10 или проверка CUDA.
- Полный набор серверного preflight: 64 passed. Дополнительные regression tests
  проверяют реальные ResNet50 output/internal gates, значения probabilities/logits,
  маски и отсутствие выдуманных gates на recovery.
- Расширенный локальный набор: 115 passed, 1 deselected. Исключён только прежний
  `test_run_history_logs_gumbel_gate_history_as_jsonl_without_duplicates`:
  он падает на `zip(strict=False)` в локальном Python 3.9. Этот отдельный JSONL
  путь отключён в pilot, не изменялся и не входит в серверный preflight.

## Дополнение: дневной режим, 6 сентября

- Добавлен `--profile daytime`: D1 dense / D2 internal-only, по 25 эпох
  (15 search + 10 recovery), один seed, один цикл, budget 10% для D2.
- 84 целевых tests passed. Дополнительно проверены передача профиля в
  дочерний процесс, точное различение 25/150 эпох, отказ от неполного job,
  отсутствие ночного порога 92% в дневном режиме и арифметика comparison.
- Реальный CPU smoke дневного профиля прошёл: D1/D2, по 1 search + 1 recovery
  на синтетических 16×16 данных. Сформированы comparison.json / comparison.md.
  D2 физически сократился до 21,193,151 параметра; это проверка исполнения,
  не результат качества на CIFAR10.
- Дневной профиль использует те же проверки/rollback/test isolation.
  GPU preflight выполняется на сервере; длинное или дневное CIFAR10 обучение
  локально не запускалось.

Работа выполнена в отдельном worktree `fix/resnet50-pruning-audit-20260905`
на основе `22c5866681680129eda04eeabefa8335e591da0e`.
Исходная открытая ветка и её незакоммиченные изменения не менялись.

## Среда и результаты

Локально: macOS, Python 3.9.6, torch 2.8.0, torchvision 0.23.0, CPU.
Нужные для пилота импорты дополнены postponed annotations; это не понижение
серверного требования Python >=3.10. CUDA/V100 в этой среде нет.

- 75 целевых tests passed: weighted/reentrant metrics, forward/backward scatter,
  legacy state load, внутренние gates без выходного selector, точная стоимость,
  floors/невалидные/немонотонные маски, best-vs-last, отсутствие доступа к test,
  rollback до/после recovery, checkpoint при SIGTERM, deadline launcher,
  четыре разрешённых конфига и прежние handoff/orchestration tests.
- Расширенный набор: 226 passed, 1 deselected, с пятью исключёнными файлами.
  Это НЕ заявление, что весь репозиторий green.
- `compileall` для src/scripts/tests и `git diff --check` прошли.
- Настоящий CPU smoke: все J1/J2/J3 прошли два цикла, по 4 эпохи / 8 optimizer
  steps, обе транзакции приняты, validation из 5 примеров с batch 4.
  Test-sentinel не сработал. Финальные физические размеры smoke:
  J1 23,547,338; J2 21,251,514; J3 15,833,760 параметров.
  Эти размеры соответствуют ДВУМ циклам; это не прогноз ночных трёх циклов.
  Данные синтетические 16×16, их accuracy не имеет экспериментального смысла.

## Почему полный suite не объявлен пройденным

Полный сбор выявил недостающие локально `torch_pruning` и `plotly`, а также
Python-3.10-only аннотации в старом `test_adaptive_lambda.py`.
Кроме того, Python 3.9 не поддерживает `zip(strict=...)` в старых gate-history
и gradient logger путях. Они не включены в фиксированный pilot; на сервере
требуется Python >=3.10.

Две проверки старых experiment-конфигов ожидают уже не совпадающие с исходным
22c5866 значения optimizer/defaults. Эти конфиги/тесты не менялись в данной работе.

Расширенный запуск исключал:

```text
tests/test_adaptive_lambda.py
tests/test_depgraph_pruning.py
tests/test_last_experiment_plotting.py
tests/test_experiment_configs.py
tests/test_gradient_norm_logging.py
test_run_history_logs_gumbel_gate_history_as_jsonl_without_duplicates
```

## Границы готовности

Проверена исполнимость фиксированного пилота, не его качество на CIFAR10,
GPU latency или способность всех jobs уложиться в ночь.
Launcher на сервере повторяет критические unit-тесты и выполняет короткий
GPU smoke перед длинным обучением; неудачный preflight останавливает запуск.
Adaptive-controller, точный resume из середины epoch и финальный test-evaluator
не реализованы; маркер полного Pro contract не добавлялся.

Старый legacy cyclic path остаётся для совместимости. Исправленный
checkpoint/budget/rollback protocol включён только в `audit_protocol=true`
и поставляемых конфигах `experiment/audit/*`. Старые threshold recipes нельзя
считать эквивалентными этому пилоту.
