# ResNet50 pruning: исправленный пилот

Исправления подготовлены в ветке `fix/resnet50-pruning-audit-20260905`
на базе `22c5866`. Основной способ получения кода — GitHub; архив не нужен.

## Запустить на сервере

Все команды ниже выполняются из корня серверного репозитория.
Нужны Python >=3.10, CUDA, зависимости проекта и pytest,
а также минимум 30 GiB свободного места. GPU и качество обучения локально
не проверялись: launcher сам запускает unit-тесты и короткий GPU smoke
до длинного обучения.

```bash
git status --short
git fetch origin
git switch --track origin/fix/resnet50-pruning-audit-20260905
.venv/bin/python scripts/launch_pruning_pilot.py \
  --output outputs/pruning_night_fixed_20260906 --data data
```

Если `git status --short` показывает незакоммиченные изменения, сначала
сохраните их; не используйте force/reset для переключения. Если локальная
ветка уже существует, вместо `git switch --track ...` выполните
`git switch fix/resnet50-pruning-audit-20260905`, затем `git pull --ff-only`.
Папка результата должна быть новой — повторное использование запрещено.
Если pytest отсутствует:
`.venv/bin/python -m pip install pytest`.

Чтобы пережить закрытие SSH, вместо команды запуска Python:

```bash
nohup .venv/bin/python scripts/launch_pruning_pilot.py \
  --output outputs/pruning_night_fixed_20260906 --data data \
  > pruning_night_fixed_20260906.log 2>&1 &
```

Не запускайте оба варианта одновременно. Для отдельной предварительной
проверки добавьте `--preflight-only` и укажите другую новую папку результата.
Обычный запуск уже включает эту проверку.

## Что будет запущено

Последовательно, на одной GPU, без test:

| Job | Gates | λ | Максимум сокращения за цикл | Минимальная исходная ширина |
|---|---|---:|---:|---:|
| J1_dense_control | bypass, без удаления | 0 | 0% | — |
| J2_output_fixed | выход Bottleneck | 0.001 | 5% физических params | 25% |
| J3_internal_fixed | только mid1 и mid2 | 0.001 | 18% физических params | 50% |

В каждом job: 3×20 search + 2×15 recovery + 60 final = 150 эпох.
У всех один случайный initializer (ноль обученных эпох), seed/split 42,
AdamW, batch 128, одинаковый протокол перезапуска optimizer/scheduler.
J1 — dense-контроль этого стадийного протокола, не непрерывное dense обучение.
Общий лимит по умолчанию 11ч45м, включая preflight; ещё до 60 секунд даётся
на сохранение при остановке. Это ограничение, не обещание, что все jobs успеют.

Четвёртый job со случайным ranking внутренних каналов — необязательный:
`--with-random-control`. Он проверяет, лучше ли learned ranking случайного
при одинаковом номинальном бюджете; фактический compute нужно сравнить отдельно.
Новый adaptive-controller отложен: это НЕ весь P0/P1 contract из AUDIT.md.

## Защиты и критерии

- Sample-weighted accuracy/CE; выбор best по validation accuracy, при равенстве — CE.
- Маска вычисляется из logits того же best.pt, из которого передаются Conv/BN.
- Бюджет считает реальный structural graph без selector-параметров; совместное
  сокращение mid1/mid2 пересчитывается после каждого удаления.
- Маски проверяются по исходным координатам: неизвестные gates, неверные индексы,
  дубликаты, повторное открытие и нарушение width floor вызывают ошибку.
- Вместо dense one-hot/einsum используется index_copy либо identity.
- Старый и предлагаемый committed graph проверяются на validation после одинаковой
  train-only BN calibration. Перед calibration проверяется равенство выходов
  структурной модели и carrier с точной бинарной маской.
- Потеря >8 п.п. до recovery или >1 п.п. после recovery отклоняет кандидат.
  Откатываются и маска, и веса. Уже потраченные эпохи не возвращаются; остаток
  идёт на fine-tuning принятой модели, дальнейшего pruning нет.
  Эти пороги — пилотные эвристики, не доказанный оптимум.
- Test dataset/loader не создаётся в training jobs. Финального test-evaluator
  в комплекте нет: сначала зафиксируйте модели и бюджеты по validation.
- NaN/Inf в loss, logits, gradients останавливают работу.
- Если J1 ниже 92% validation, последующие jobs не стартуют. Это sanity gate.
- Незавершённый job не считается 150-эпоховым. При TERM сохраняется
  interrupted.pt с optimizer/scheduler/RNG и отметкой частичной эпохи, если
  остановка застала training. Автоматического точного resume с середины batch/
  DataLoader-worker state здесь НЕТ. Best/last сохраняются атомарно.

Идеальные цели — около 20.19M params для J2 и 12.98M для J3; floors,
дискретность и rollback могут помешать. `parameter_target_met` допускает 1%
от идеальной арифметической цели для дискретности; это не оценка качества.
Нельзя считать job успешным только потому, что сохранена accuracy.

## Где смотреть результат

- `nightly_status.json`: завершённые/пропущенные jobs, ошибки.
- `J*/pilot_state.json`: решения, причины отката, validation, реальные params/MACs.
- `J*/global_history.csv`: глобальные эпохи, weighted validation, λ used/next, LR.
- `J*/deployment.pt`: принятая структурная модель, маска, hash и validation;
  при откате checkpoint может быть выбран раньше конца, но все затраты учтены.
- `J*/cycle_*/.../checkpoints/{best,last,interrupted}.pt`: стадийные checkpoint.
- `provenance.json`: hashes исходников, среда и GPU; `resolved_config.yaml`: конфиг job.

MACs охватывают Conv/Linear; BN, ReLU, pooling, residual add и scatter перечислены
как неучтённые операции, а не объявлены бесплатными. Финальная GPU latency
измеряется с warmup и синхронизацией для batch 1/128. Это измерение финального
графа, не обещание ускорения и не полный FLOP count.

## Локальная проверка / интеграция

`scripts/smoke_pruning_pilot.py` обучает реальные ResNet50 на маленьких
синтетических данных: три варианта, два цикла, неполный valid batch и test-sentinel.
Это проверка корректности исполнения, не качества на CIFAR10.

Review исправлений доступен через `git diff 22c5866..HEAD` в этой ветке.
Для обычного запуска не нужны ZIP или patch. Если всё же нужен автономный
снимок, `scripts/build_pruning_pilot_bundle.py` создаёт ZIP с исходниками
и `CHANGES_FROM_22c5866.patch`; это необязательный способ передачи.
