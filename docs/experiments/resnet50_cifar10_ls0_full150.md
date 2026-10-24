# Один полный ResNet50/CIFAR10 запуск после repro20

Одна новая модель обучается с исходного zero-epoch initializer: 60 эпох поиска
маски и 90 эпох физического recovery. Всего 150 новых training epochs.
Диагностические повторы остаются отдельными результатами; эта команда начинает
новое обучение с эпохи 1. Причина различия repro20 пока не установлена, и полный
запуск не означает, что воспроизводимость уже исправлена.

Поиск использует тот же рецепт, что repro20: seed42, LS=0, adaptive lambda,
soft_drop=0.00025, hard_drop=0.0005, min_keep_ratio=0.08, tau=0.5,
warmup10 при lambda=0.001, log_step=auto (0.9210340371976183), cosine60.
Настройки CUDA/cuDNN/TF32 команда не меняет. Полная batch-диагностика repro20
в этой команде не включается.

Физические кандидаты выбираются по validation только среди эпох 31–60 после
200 train-only BN batches для каждого. Это 6,000 дополнительных train forward
batches, не эпох обучения и не шагов оптимизатора. Одна ветвь recovery наследует
выбранные веса, использует LS=0, BN200 перед обучением, fresh AdamW и cosine90.
В физической модели нет гейтов; adaptive controller сохраняется как состояние
поиска. Размер модели измеряется по факту.

Итоговый checkpoint выбирается по validation. Затем команда оценивает
зафиксированную модель на всех 10,000 official test примерах без обучения и
обновлений BatchNorm. Итоговое сравнение с прежними результатами — exploratory.
При отсутствии допустимого checkpoint сохраняется явный quality_feasible=false;
факт завершения не означает выполнения ограничения по качеству.

Нужны уже перенесённые original reference и initializer в
`outputs/baseline/one_shot_dense_reference_seed42`. Обе фазы проверяют входы;
хеши истории47b56d и initializer82e723 закреплены в manifest. Дополнительного
обучения dense-reference нет. Каждый запуск требует новый output-каталог.

## Проверка без обучения

Из корня репозитория:

```bash
.venv/bin/python scripts/launch_resnet50_cifar10_ls0_full150.py --validate-only
```

Для проверки существующих входов, GPU и свободного места используйте
`--preflight-only` вместо `--validate-only`. Штатный preflight требует 55 GiB
свободного места. По завершении фаз runner удаляет вложенные epoch checkpoints,
сохраняя выбранные checkpoints и deployments, как в предыдущих полных запусках.

## Запуск

```bash
nohup .venv/bin/python -u scripts/launch_resnet50_cifar10_ls0_full150.py \
  > resnet50_cifar10_ls0_full150_v1.log 2>&1 &
tail -f resnet50_cifar10_ls0_full150_v1.log
```

Результаты: `outputs/runs/resnet50_cifar10_ls0_full150_v1/`.
`study_state.json` показывает этапы и ошибки; `validation_results.csv` содержит
зафиксированную validation-оценку, `final_results.csv` — также official test.
В manifest ровно один профиль; после его оценки runner завершает работу.
