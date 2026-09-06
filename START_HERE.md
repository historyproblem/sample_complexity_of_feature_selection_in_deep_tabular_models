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

## Короткий дневной пилот

На уже полученной ветке сначала выполните `git pull --ff-only`. Вместо ночной
команды запустите:

```bash
.venv/bin/python scripts/launch_pruning_pilot.py --profile daytime \
  --output outputs/pruning_day_20260906 --data data
```

Это два запуска без повторов seed: D1 dense и D2 internal-only pruning.
У каждого 15 search + 10 recovery = 25 эпох, только один цикл.
Общий случайный initializer и split/seed 42, test выключен.
У D2 удаляется до 10% физических параметров за единственное решение:
ориентир ~21.2M вместо 23.55M, floor внутренних ширин 50%.
Выходные каналы не прунятся. λ=0.001, entropy=0, adaptive-controller выключен.
Предварительные тесты/GPU smoke и rollback сохраняются.

Лимит дневного профиля — 2 часа на всё, включая preflight, плюс до 60 секунд
на остановку. Это верхняя граница, не обещание времени завершения.
`--hours` меняет только лимит, а не число эпох. Незавершённый job не
попадает в сравнение как полноценный результат. Требование 92% для dense
здесь отключено: за 25 эпох нельзя требовать качества 150-эпохового запуска.

После каждого завершённого job появляются `comparison.json` и
`comparison.md` с validation, разницей относительно dense, params/MACs
и решениями/отказами pruning. Для совместного анализа достаточно прислать
`comparison.json` и оба `D*/global_history.csv`.
Логи первых эпох — `D1_dense_control.log` / `D2_internal_fixed.log`
в папке результата; технический preflight пишет отдельные логи.

Что проверяем:

- Если даже dense плохо учится, сначала проверяем обучение/данные, а не
  приписываем проблему pruning.
- Если committed mask резко портит validation или включается rollback,
  проверяем распределение удалений и gate probabilities перед усилением pruning.
- Если качество восстановилось, но physical target не достигнут, это не
  успешное сокращение модели.
- Если target достигнут и отставание от dense невелико, это повод продолжить
  более длинный pilot, а не доказательство качества обученного ranking.
  Random ranking при том же бюджете — следующий контроль, когда он понадобится.

Один seed и 25 эпох дают раннюю диагностику, не окончательный выбор архитектуры.
Опциональный `--with-random-control` добавляет D3 с тем же коротким бюджетом;
в дневной команде выше этот дополнительный job не включён.

## Что будет запущено ночью (профиль по умолчанию)

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
- В ночном профиле: если J1 ниже 92% validation, последующие jobs не стартуют.
  Это sanity gate; он не применяется к короткому дневному профилю.
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
- `comparison.json` / `comparison.md`: компактное сравнение завершённых jobs.
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
