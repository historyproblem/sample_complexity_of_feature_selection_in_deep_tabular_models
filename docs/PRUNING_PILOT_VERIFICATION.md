# Проверки от 5 сентября 2026

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
