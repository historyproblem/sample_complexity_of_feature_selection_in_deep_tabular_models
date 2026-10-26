# Восемь повторов одного pre-BN конфига

Серия повторяет `large_15m_keep001` из
[четырёх запусков](resnet50_cifar10_ls0_prebn_repro4.md) восемь раз.
Каждый повтор запускает собственный search с нуля и собственный recovery.
Результаты предыдущих повторов не используются для обучения следующих.

Во всех восьми повторах одинаковы:

- soft/hard: **0.00025 / 0.0005**, `min_keep_ratio=0.001`;
- L1, adaptive lambda с auto-step, entropy0, LS0 и **seed42**;
- прежние начальные веса и dense reference, закреплённые хешами;
- 60 эпох search, затем 90 эпох recovery: **150 на модель, 1200 суммарно**;
- отбор на эпохах 31–60: минимум физических параметров среди прошедших
  validation-порог; если таких нет — лучшая validation в том же окне;
- отсутствие дополнительной BN-калибровки в search, включая selected export;
- BN200 на train после импорта surviving weights в recovery;
- fresh AdamW и fresh cosine90 в recovery, сохранение состояния контроллера.

Меняются только идентификаторы повторов `01`–`08` и каталоги результатов.
Search/recovery phase configs повторно используются без изменений из prebn_repro4.
Это проверка воспроизводимости **при одном и том же seed**, а не оценка разброса
по разным seeds. Настройки CUDA-детерминизма остаются прежними. Различия между
повторами возможны, но заранее не гарантируются. Целевого размера 5M или 15M
в отборе нет; сохранённое имя `large_15m` — исторический ориентир.

Сначала фиксируются validation-результаты всех восьми моделей, затем выполняется
frozen official test на 10 000 примерах каждой. На test нет обучения или
обновлений BN. Все результаты сохраняются: серия не выбирает победителя по test.
Сравнение исследовательское, поскольку исторические test-результаты известны.

## Проверка и запуск

Команды выполняются из корня репозитория на GPU-сервере:

```bash
git fetch origin
git switch feature/resnet50-cifar10-prebn-repro4
git pull --ff-only
.venv/bin/python scripts/launch_resnet50_cifar10_ls0_prebn_repeat8.py --validate-only
.venv/bin/python scripts/launch_resnet50_cifar10_ls0_prebn_repeat8.py --preflight-only
```

`--validate-only` проверяет все 16 конфигов фаз на CPU. `--preflight-only`
проверяет также реальные reference/initializer, GPU и свободное место без обучения.

```bash
nohup .venv/bin/python -u scripts/launch_resnet50_cifar10_ls0_prebn_repeat8.py \
  > resnet50_cifar10_ls0_prebn_repeat8_v1.log 2>&1 &
tail -n 100 -f resnet50_cifar10_ls0_prebn_repeat8_v1.log
```

Повторы идут последовательно. `Ctrl+C` в `tail` прекращает просмотр,
а обучение продолжает работать. Каталог серии:
`outputs/runs/resnet50_cifar10_ls0_prebn_repeat8_v1/`.
Для каждого повтора создаются отдельные `search_large_15m_keep001_repeat_NN/`
и `recovery_large_15m_keep001_repeat_NN/`; итоговая таблица — `final_results.csv`.
В ней восемь строк с фактическими параметрами, validation и official test.

Для проверки и переиспользования уже завершённых повторов есть
`--resume-completed`. Он сохраняет привязку recovery к search того же повтора,
проверяет настройки и хеши. Частично обученный повтор автоматически не начинает
обучение заново. Ошибка одного повтора записывается, остальные продолжаются;
при наличии ошибок итоговый код завершения ненулевой.

По умолчанию вложенные checkpoints законченной фазы очищаются после проверки,
выбранные root-checkpoints сохраняются. `--keep-nested-checkpoints` сохраняет
полные истории checkpoints и требует больше места.

## Локальные проверки

```bash
.venv/bin/python -m pytest -q tests/test_resnet50_ls0_prebn_repeat8.py
```
