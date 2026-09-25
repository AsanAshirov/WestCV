# CLAUDE.md — контекст проекта WestCV (команда Antigradient)

Общаемся с командой **по-русски**. Команда из 3 человек, отборочный этап WIUT Hackathon 2026, трек Computer Vision.
**Дедлайн: воскресенье, 27 сентября 2026, 23:59 по Ташкенту.** Оценивается коммит с тегом на момент дедлайна. Финальный тег ставим к 21:00, после этого ничего не пушим.

## Задача (кратко)
- **Part A:** `detect_events(video_path) -> [[start_sec, end_sec, label], ...]`, 14 классов. Метрика: macro-F1 по tIoU 0.3/0.5/0.7. Класс, который мы предсказали, но которого нет в тесте, добавляет в среднее 0.
- **Part B:** `RiskEstimator.reset(meta)` / `step(frame, t_sec) -> float`: причинная вероятность того, что авария начнётся в ближайшие 5 с.
- `M = 0.7*A + 0.3*B` (`M = A`, если в тесте нет аварий). `Отбор = 0.6*M + 0.25*Сайт + 0.15*Код`.
- Лимиты: T4 16 ГБ, 8 CPU, без интернета, **Part A + Part B ≤ 3× длительности видео** (иначе видео пустое), веса ≤ 5 ГБ.
- Полные материалы: `docs/WestCV_plan_RU.pdf`, `WIUT_CV_Track_analysis_RU.md`, `WIUT_toolkit_RU.md`. Пошаговая инструкция для команды: **`docs/RUNBOOK_RU.md`**.

## Проверенные факты
- Сэмплы лежат локально в `DataSets/` (в git не попадают): C3896, C3897, C3902, C3905. Формат: 3840×2160, 29.97p, H.264 High 4:2:2 10-bit, ~140 Мбит/с. Около 18.4 мин, ~20 ГБ. C3897 и C3902 совпадают по размеру до байта — сверить хеши.
- **`camera.md` в задании больше нет.** Геометрию сцены (переходы и т. п.) рисуем сами в labelme, остальное учится по движению в самом видео.
- Время всегда `frame_index / CAP_PROP_FPS`, как в харнессе. Fps не хардкодить: в PDF написано 25, сэмплы 29.97.
- Харнесс: `solution.py` импортируется до таймера, поэтому модели грузим при импорте. Цикл Part B харнесса сам читает каждый 4K-кадр через `cv2.read()`, это ≈0.85–1.0× длительности на 8 vCPU и до ≈2× на g4dn. Превышение бюджета обнуляет и A, и B. Исключение в `step` → `risk=[]`, события сохраняются (PDF формулирует иначе, на это не рассчитываем).
- NVDEC на T4 не декодирует H.264 4:2:2. Декод только на CPU.

## Архитектура (реализована, `src/trafficwatch/`)
Решение **без обучения**: готовый детектор YOLO26m (веса COCO, лежат в `weights/`), собственный ByteTrack-трекер, правила на траекториях. Внешние датасеты не нужны.

| Файл | Что делает |
|---|---|
| `solution.py` | Тонкая обёртка: `CLASSES`, `detect_events`, `RiskEstimator`; прогрев моделей при импорте |
| `env.py` | Офлайн-переменные (до импорта torch/ultralytics), сиды, детерминизм |
| `config.py` + `configs/pipeline.yaml` | **Все пороги здесь.** Единицы — секунды и BS (размер бокса = sqrt(w·h); у машины 1 BS ≈ 2–3 м) |
| `video.py` | Метаданные как в харнессе; один последовательный проход: ffmpeg (imageio-ffmpeg) масштабирует сам и работает в отдельном процессе; запасной путь — cv2 grab/retrieve |
| `detector.py` | YOLO → 4 суперкласса (vehicle, two_wheeler, person, animal), NMS внутри суперкласса |
| `tracker.py` | ByteTrack-lite: две стадии, время жизни в секундах, без `fuse_score`. Общий для A и B |
| `perception.py` | Декод + детекция с дедлайном → `Perception` (кэшируется в `.npz`); трекинг |
| `tracks.py` | Разрез треков на скачках ID, склейка стоящих, сглаживание, скорость в BS/с, курс |
| `scene.py` | `configs/scene.json` (необязателен) + карты по видео: маска дороги (где ездят машины), поле направлений |
| `rules/*.py` | По модулю на класс: `stopped_vehicle`, `jaywalking`, `wrong_way`, `congestion`, `accident` |
| `postprocess.py` | union → gap-merge → min-dur → санитайзер (3 знака, end ≤ floor(duration), без перекрытий) |
| `pipeline.py` | Part A: бюджет `T_A = (3 − 1.25·h − 0.25)·dur` по замеру цикла харнесса, затем perception → analyze |
| `risk.py` | Part B: своя детекция YOLO26n@640 ~10 Гц, TTC пар, два канала (≤0.4999 ранжирование, ≥0.5 тревога ≤ 8 с). Без CUDA выключается. Следит за дедлайном |

Классы, которые не реализованы (`near_miss`, `red_light`, `stop_line`, `failure_to_yield`, `solid_line_crossing`, `illegal_turn`, `illegal_u_turn`, `road_obstacle`, `fire_smoke`), не предсказываются. Добавлять класс только после проверки ложных срабатываний на всех 4 сэмплах: ложный класс стоит дороже пропущенного.

## Команды
```bash
python -m pytest -q tests                                              # 15 тестов, ~5 с, без видео
python tools/bench_decode.py DataSets/C3905.MP4 --seconds 20           # бюджет времени на этой машине
python tools/cache_perception.py --videos DataSets --out cache         # детекции один раз
python tools/run_rules.py --cache cache --gt dev_labels/dev_gt.json    # правила + официальный скор, секунды
python tools/render_review.py DataSets/C3896.MP4 --cache cache --out review --gt dev_labels/dev_gt.json
python tools/csv_to_gt.py dev_labels/labels.csv --videos DataSets --out dev_labels/dev_gt.json
python tools/make_reference_frame.py DataSets/C3896.MP4 --out geometry/ref_C3896.png   # для labelme
python run_submission.py --videos DataSets --out predictions_samples.json --team Antigradient
python evaluate.py --pred predictions_samples.json --validate-only
```
Альтернативный конфиг: `TW_CONFIG=path/to.yaml` или `run_rules.py --config`.

## Правила, которые нельзя нарушать
- `run_submission.py` и `evaluate.py` **не модифицировать**.
- Только open-weights, никаких API на инференсе. Никаких загрузок в рантайме: веса по абсолютному пути, `env.py` импортируется первым.
- `RiskEstimator` видит только полученные кадры. Из Part A в Part B передаётся только время старта видео (`pipeline.RUN_CLOCK`), никаких данных.
- Детерминизм: сиды, последовательный декод, никаких решений по настенным часам, кроме бюджетного тормоза.
- Не использовать записи с той же камеры, полученные другими путями. Все данные и лицензии перечислены в README.
- Видео, кэши, review-видео не коммитить (`.gitignore`). Веса `weights/yolo26m.pt` и `yolo26n.pt` закоммичены намеренно.
- Репозиторий под AGPL-3.0 (Ultralytics). Не обучать Ultralytics-модели на данных с NC-лицензией.

## Что делать дальше (по приоритету)
- [ ] Прогнать `pytest`, `bench_decode.py` и полный `run_submission.py` на реальных сэмплах. Записать `total_sec / duration` для каждого видео.
- [ ] `cache_perception.py` на всех сэмплах, `render_review.py`, посмотреть, что видит конвейер.
- [ ] Нарисовать переходы в labelme → `configs/scene.json` (без них `jaywalking` выключен).
- [ ] Быстрая dev-разметка в `dev_labels/labels.csv` → `dev_gt.json`.
- [ ] Подбор порогов через `run_rules.py`; выключить классы с ложными срабатываниями.
- [ ] Прогон на Kaggle T4, запас по времени ≥ 20%. Сгенерировать там `predictions_samples.json`.
- [ ] README: команда и кто что сделал. Тег `v1.0` к 21:00 воскресенья.
- [ ] Сайт с live-демо (25% отбора) — параллельно, отдельный человек.
- [ ] Перед сдачей убрать `research/` или вынести из репозитория (рубрика «no dead code»).
