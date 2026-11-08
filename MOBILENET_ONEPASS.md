# MobileNetV2: однопроходный поиск и обучение архитектуры

Ветка `feature/mobilenet-onepass-thr05-gap05-1pp` основана на `mobilenetv2`
(`6501c07`) и содержит совместимые модели и каналовый драйвер. Одного конфига
поверх старого `develop` для запуска недостаточно.

Все команды ниже выполняются из корня репозитория
`sample_complexity_of_feature_selection_in_deep_tabular_models`.
Каталог загрузки — **`data/` внутри репозитория**, без ведущего `/`.

## Новый запуск: допуски λ 7/8 п.п., без cap

`configs/mobilenet_onepass_thr05_gap7_8pp.yaml` задаёт `soft_drop=0.07`,
`hard_drop=0.08`, порог каналов 0.5 и `threshold_max_param_fraction: null`.
Один профиль автоматически выполняет 30 эпох поиска и 45 эпох обучения
полученной физической архитектуры, затем test checkpoint, выбранного по validation.
Начальная λ 1e-4, warmup 3, адаптивный шаг, ImageNet V2, label smoothing 0,
batch 128 и workers 0 сохранены. Используется прежний архивный reference.

```bash
git pull --ff-only
mkdir -p outputs/logs
nohup .venv/bin/python -u scripts/run_mobilenet_onepass.py \
    --config-name mobilenet_onepass_thr05_gap7_8pp \
    --baseline-archive outputs/baselines/mobilenet_baseline_partial.tar.gz \
    > outputs/logs/mobilenet_onepass_thr05_gap7_8pp_nocap.log 2>&1 < /dev/null &
tail -f outputs/logs/mobilenet_onepass_thr05_gap7_8pp_nocap.log
```

Результаты отдельные: `outputs/runs/mobilenet_onepass_thr05_gap7_8pp/`.
Это новый поиск с новой инициализацией λ, без продолжения прежнего запуска.

В присланном логе прежнего 6/7 п.п. поиска λ снижалась на эпохах 4–16
(13 раз), не менялась на 17-й, росла на 18–30-й (13 раз); первые три эпохи — warmup.
Снижение умножало λ на 0.8, рост с 19-й эпохи — примерно на 2.4414 из-за boost.
Коэффициент 0.385186 после эпохи 30 уже не использовался в обучении этой модели.
На той же записанной кривой validation переход к 7/8 п.п. сдвинул бы начало роста
с 18-й на 17-ю эпоху. Это условный пересчёт по прежним округлённым метрикам,
а не прогноз нового обучения или обещание заметного сжатия.

## Новый конфиг: допуски λ 6/7 п.п.

`configs/mobilenet_onepass_thr05_gap6_7pp.yaml` использует порог каналов 0.5
**без лимита cap** (`threshold_max_param_fraction: null`), batch size 128,
workers 0, адаптивную λ, entropy 0 и label smoothing 0. Допуски
заданы как `soft_drop=0.06`, `hard_drop=0.07`: λ увеличивается при отставании
до 6 п.п., сохраняется между 6 и 7 п.п. и уменьшается при отставании свыше 7 п.п.
Reference остаётся поэпоховой историей выбранного архива, не постоянным best.

Одна команда выполняет полный новый запуск:
**30 эпох поиска → выбор и физическое удаление каналов → 45 эпох обучения
полученной архитектуры → test выбранного по validation checkpoint**.
Recovery начинается с ImageNet-весов сохранившихся каналов, как прежде.
Новый search начинает адаптивную λ с 1e-4; старый search не продолжается.
Всего 75 новых эпох. Новый baseline не обучается.

После обновления ветки из корня репозитория:

```bash
git pull --ff-only
mkdir -p outputs/logs
nohup .venv/bin/python -u scripts/run_mobilenet_onepass.py \
    --config-name mobilenet_onepass_thr05_gap6_7pp \
    --baseline-archive outputs/baselines/mobilenet_baseline_partial.tar.gz \
    > outputs/logs/mobilenet_onepass_thr05_gap6_7pp_nocap.log 2>&1 < /dev/null &
tail -f outputs/logs/mobilenet_onepass_thr05_gap6_7pp_nocap.log
```

С этим `--config-name` по умолчанию запускается **один профиль без cap**:
удаляются каналы с `p_open < 0.5`, с сохранением хотя бы одного канала
в каждой группе. Launcher передаёт `threshold_max_param_fraction=null`.
Результаты записываются отдельно в `outputs/runs/mobilenet_onepass_thr05_gap6_7pp/`.
Имя профиля заканчивается на `_nocap`, поэтому результаты прежнего запуска
с cap 0.1 сохраняются в своей директории.
Посмотреть команду полного цикла без обучения можно через `--dry-run`.

Без `--config-name` сохраняется прежняя серия с допусками 0.5/1 п.п. и шестью cap.
Инструкции подготовки данных и окружения ниже применимы ко всем вариантам.

Launcher передаёт `-u` также дочернему процессу обучения, чтобы строки Python
не задерживались в буфере при записи в файл. Это действует для новых процессов;
уже запущенное обучение менять или повторять ради вывода не требуется.
Строка итогов эпохи появляется после train и validation. В измеренном первом
search эпоха занимала в среднем 11 минут 49 секунд; отдельные эпохи — более12минут.

Если вывод долго не обновляется, сначала проверьте процессы и GPU. После
Ctrl+C из `tail` эти команды только читают состояние:

```bash
ps -eo pid,ppid,stat,etime,pcpu,args | grep -E '[r]un_mobilenet_onepass.py|[c]yclic_train.py'
nvidia-smi
```

`history.csv` обновляется после завершения эпохи. Отсутствие её первой строки
до конца эпохи и тишина stdout сами по себе не подтверждают зависание.

## Получить ветку

```bash
git fetch origin
git switch feature/mobilenet-onepass-thr05-gap05-1pp
```

## Собрать данные и установить зависимости

Отдельная команда для сборки восьми загруженных частей в один ZIP:

```bash
cat data/tiny-imagenet-200.zip.part-{a,b,c,d}.sub-{a,b} > data/tiny-imagenet-200.zip
```

Общая подготовка также умеет собирать части сама, поэтому предыдущая команда
не обязательна. Она проверяет SHA256 и CRC ZIP, распаковывает датасет,
проверяет 200 классов, 100000 train и 10000 размеченных official-val изображений,
устанавливает зависимости и выполняет настоящую CUDA-проверку:

```bash
bash scripts/setup_mobilenet_onepass.sh
```

Ожидаемый SHA256 архива:
`6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821`.
Датасет после распаковки: `data/tiny-imagenet-200/`. Готовые совпадающие ZIP
и полная распаковка переиспользуются; повреждённые или частичные данные
потребуют исправления вместо незаметной перезаписи.

Зависимости повторяют ранее проверенный серверный V100-стек:
`torch==2.6.0+cu126`, `torchvision==0.21.0+cu126`, `protobuf>=3.20.3,<6`
и полный набор библиотек из `requirements-pruning-v100.txt`. Установка
и все запуски идут через `.venv/bin/python`, чтобы пакеты не попадали
в другую Conda/Python-среду. `requirements.txt` и `setup.py` также исправлены,
чтобы не требовать Torch 2.10 / torchvision 0.25. Если torchaudio уже установлен,
setup выравнивает его версию с Torch; для нового окружения он не добавляется.

Нужен Python ≥3.10; ранее на сервере использовался Python 3.12.9.
Если `.venv` отсутствует, setup создаёт его базовым `python3`; при необходимости
выберите интерпретатор через `PRUNING_BOOTSTRAP_PYTHON=python3.12`.
Проверка включает `pip check`, матричное умножение на CUDA, torchvision NMS
и наличие `sm_70` для V100. Обучение setup не запускает.

Только сборка ZIP без установки или распаковки, если `.venv` уже существует:

```bash
.venv/bin/python scripts/prepare_tinyimagenet.py --assemble-only
```

## Подготовить выбранный пользователем baseline

Загрузите `mobilenet_baseline_partial.tar.gz` из локального Downloads в корень
серверного репозитория. Архив имеет размер 54 823 040 байт и SHA256
`7ada53ae4c56902dcfc48d3625273d66955dd8b9ca7702727d9854d342bf9cfd`.
Git этот архив не переносит.

```bash
.venv/bin/python scripts/prepare_mobilenet_reference.py
```

Импортируются только `history.csv` и `config_resolved.yaml`, с проверкой SHA256.
Checkpoint не загружается в модель и не нужен для адаптивной λ. Исходная история
сохраняется целиком: 86 эпох из запланированного обучения на 150 эпох,
с cosine `T_max=150`. Во время поиска на 30 эпох контроллер сопоставляет
её эпохи 1–30. Это явный выбор
пользователя; нового reference с cosine на 30 эпох здесь нет. Новое обучение
baseline убрано; при отсутствии архива launcher остановится, а не начнёт
тренировать другой reference.

## Запустить обучение

После успешного setup и загрузки архива:

```bash
mkdir -p outputs/logs
nohup .venv/bin/python -u scripts/run_mobilenet_onepass.py \
    > outputs/logs/mobilenet_onepass_thr05_gap05_1pp.log 2>&1 < /dev/null &
```

Посмотреть ход выполнения:

```bash
tail -f outputs/logs/mobilenet_onepass_thr05_gap05_1pp.log
```

Launcher проверяет и импортирует выбранный архив без обучения baseline. Затем
последовательно запускает **шесть** независимых профилей с cap
`0.1, 0.2, 0.3, 0.4, 0.5, 0.6`. Для каждого:

- Один цикл: **30 эпох поиска + 45 эпох структурного recovery**, всего 75.
- DataLoader использует `num_workers=0`: загрузка в основном процессе без
  межпроцессных очередей в ограниченной `/dev/shm` серверного контейнера.
  Batch size 128 сохранён.
- Порог открытия для кандидатов: `p_open < 0.5`.
- Адаптивная λ включена: `soft_drop=0.005`, `hard_drop=0.01`.
- Начальная λ `1e-4`, warmup 3, окно 3, прежний closure-rate boost.
- Внутренние гейты включены, entropy 0, label smoothing 0 в обеих фазах.
- Поиск и recovery заново стартуют из ImageNet V2. Recovery не имеет гейтов;
  соответствующая регуляризация там отсутствует. Следующего search нет.

Новое обучение составляет **75 эпох на профиль**, в пределах 150-эпохового
ограничения. Исторические 86 эпох отдельного baseline учитываются отдельно как
переиспользуемый reference; его веса не являются начальной точкой моделей. В `cyclic_summary.json`
исторический reference отражается отдельно как `one_time_baseline_epochs=86`;
новые training epochs профиля остаются равны75.
Лимит cap — максимум стоимости новых удалений, он не гарантирует заданный
физический размер модели.

Данные делятся на 90000 train / 10000 внутренней validation по seed42;
размеченная официальная val-часть используется как test. Выбор checkpoint
происходит по validation; финальные метрики берутся из test выбранного
checkpoint. Это один цикл исторического iterative-драйвера, с прежним отбором
каналов по последним validation-метрикам поиска; правило нового ResNet
Algorithm 1 сюда не переносится.

Результаты находятся в `outputs/runs/mobilenet_onepass_thr05_gap05_1pp/`:
импортированная история reference и её provenance, отдельные директории cap, `cyclic_summary.json`,
истории и checkpoints. MLflow использует локальную SQLite-базу репозитория.
ImageNet V2 weights скачиваются в Torch cache при первом построении модели,
если их там ещё нет.

Печать точных команд без обучения и без CUDA:

```bash
.venv/bin/python scripts/run_mobilenet_onepass.py --dry-run
```

Один профиль, например cap 0.4:

```bash
.venv/bin/python scripts/run_mobilenet_onepass.py --caps 0.4
```

Launcher блокирует одновременный повторный запуск и переиспользует только
завершённые собственные training stages с совпадающим config/code fingerprint.
Путь к архиву при необходимости задаётся через `--baseline-archive /path/to/archive`.
Если stage частично обучен, он останавливается, не добавляя новые эпохи.

## Повтор после DataLoader bus error

В серверном контейнере первый cap0.1 упал до завершения первой эпохи:
`DataLoader worker ... Bus error`, с сообщением о shared memory.
Конфиг теперь явно задаёт `num_workers=0`, сохраняя batch size128, эпохи
и параметры метода. Загрузка выполняется в основном процессе.

После `git pull --ff-only` сохраните всю директорию неудачной попытки отдельно:

```bash
failed_run=outputs/runs/mobilenet_onepass_thr05_gap05_1pp/mobilenet_onepass_thr0.5_soft0.005_hard0.01_cap0.1
mv "$failed_run" "${failed_run}_failed_shm"
```

Затем повторите команду launcher с
`--baseline-archive outputs/baselines/mobilenet_baseline_partial.tar.gz`.
Импортированный reference переиспользуется. Перенос сохраняет артефакты ошибки
и освобождает путь для новой попытки; без переноса сработает защита от повтора
частично выполненного stage. Первый упавший процесс должен уже завершиться
перед переносом. Лог ошибки не исключает отдельных batch updates до падения;
не считайте старую попытку завершённым обучением или recovery.

## Локальная проверка перед публикацией

- Проверки сборки данных, импорта reference и оркестрации: PASS (см. тесты).
- Сами восемь загруженных частей: 248100043 байта, SHA256 совпал; ZIP CRC PASS.
- Настоящий CLI успешно печатает Hydra-конфигурацию без обучения.
- 75 существующих проверок MobileNet и каналового цикла прошли. Ещё 9 проверок
  исходной ветки имеют устаревшие ожидания ImageNet V1 / старого resize preset;
  связанные исходники, конфиги и тесты не изменялись этим запуском.
- Локальная среда проверки — CPU Torch 2.8 / torchvision 0.23. Установка
  V100-зависимостей и GPU-обучение выполняются только пользователем на сервере;
  setup проверяет именно серверные wheels и CUDA перед запуском.
