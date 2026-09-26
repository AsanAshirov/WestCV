# RUNBOOK: как запустить, настроить и сдать (команда Antigradient)

Дедлайн: **воскресенье, 27.09.2026, 23:59 (Ташкент)**. Финальный тег ставим к **21:00**.

## 0. Решение и почему без обучения

**Модель = готовый детектор + трекер + правила.** Так же советуют организаторы: «детектор и трекер дают траектории; большинство классов — правила на траекториях и разметке сцены».

| Часть | Что используем | Обучается? |
|---|---|---|
| Детекция машин, людей, мотоциклов | YOLO26m, готовые веса COCO (`weights/yolo26m.pt`, 44 МБ, в репозитории) | Нет, готовые веса |
| Трекинг | Свой ByteTrack (`src/trafficwatch/tracker.py`) | Нет |
| Карта дороги и направления полос | Учатся **по самому видео**: где ездят машины и куда | Да, без разметки, на каждом видео |
| 5 классов (`stopped_vehicle`, `jaywalking`, `wrong_way`, `congestion`, `accident`) | Правила в `src/trafficwatch/rules/` | Пороги подбираем на своей разметке |
| Part B (риск аварии) | YOLO26n + TTC между парами | Нет, пороги в конфиге |

Почему не обучаем за 48 часов:
- COCO-детектор уже хорошо видит машины и людей; дообучение без разметки даст мало и съест день.
- Обучаемая модель аварий (VideoMAE и т. п. на ACCIDENT) — это 6–10 часов работы плюс риск по времени инференса, а точные границы события она всё равно не даёт.
- Главный рычаг качества — **пороги и границы, настроенные на своей разметке**. Это и есть наша «тренировка» (§6–7).

**Внешние датасеты качать не нужно.** Опциональные варианты, если останется человек и время, — в §9.

## 1. Установка

### Windows (ноутбук)
```bat
cd WestCV
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
python -m pytest -q tests
```
Без `uv`: `py -3.11 -m venv .venv` и `pip install -r requirements.txt`. На Python 3.14 не ставить: для torch 2.14 нет колёс.

### Linux / Kaggle
```bash
pip install -r requirements.txt
python -m pytest -q tests      # должно быть 15 passed
```

Веса уже в репозитории. Проверить их: `bash weights/download.sh` — докачивает недостающие и сверяет SHA-256.

## 2. Данные

- Сэмплы положить в **`DataSets/`** в корне репозитория: `DataSets/C3896.MP4` и т. д. Папка в `.gitignore`.
- Сверить C3897 и C3902 (могут быть дубликатом): `certutil -hashfile DataSets\C3897.MP4 SHA256` на Windows или `sha256sum` на Linux.

## 3. Первый прогон (≈30 минут) — замер времени

```bash
python tools/bench_decode.py DataSets/C3905.MP4 --seconds 20
python run_submission.py --videos DataSets --out predictions_samples.json --team Antigradient
python evaluate.py --pred predictions_samples.json --validate-only
```

- `bench_decode.py` печатает **h** — сколько занимает чтение кадров самим харнессом в долях длительности — и сколько на Part A остаётся из лимита 3×.
- В `predictions_samples.json` в блоке `log` посмотреть `total_sec / duration` для каждого видео. **Цель — не больше 2.5× на T4.**
- Если время впритык, см. §8.

## 4. Кэш детекций (один раз, дальше всё за секунды)

```bash
python tools/cache_perception.py --videos DataSets --out cache
```

Считать на машине с GPU (ноутбук с GTX 1650 подойдёт, Kaggle T4 быстрее). В `cache/*.npz` сохраняются все детекции. Трекинг и правила потом перезапускаются из кэша за секунды, без повторного декодирования 4K. После изменения секций `decode` или `detector` в конфиге кэш нужно пересчитать с `--force`.

## 5. Геометрия сцены (≈20 минут) — нужна для `jaywalking`

Дорогу и направления полос конвейер учит сам. Руками рисуем только **пешеходные переходы**: без них любой пешеход на переходе был бы «jaywalking», поэтому класс выключается сам.

```bash
python tools/make_reference_frame.py DataSets/C3896.MP4 --out geometry/ref_C3896.png
labelme geometry/ref_C3896.png --labels configs/scene_labels.txt --validate-label exact --output geometry/
python tools/annotation/labelme_to_scene.py geometry/ref_C3896.json configs/scene.json
```

- `pip install labelme` ставить в **отдельный** venv: labelme тянет свой OpenCV.
- В labelme рисовать полигоны: `crosswalk` (обязательно, если есть переходы), по желанию `sidewalk`, `island`, `parking` и `carriageway`.
- Предупреждение `missing labels: lane_dir, stop_line` игнорировать.
- **Если переходов в кадре нет вообще**, поставить `jaywalking.require_crosswalks: false` в `configs/pipeline.yaml`.
- Сессии днём и вечером могли сниматься с разной установкой штатива. На review-видео (§6) проверить, что переходы совпадают на всех сэмплах. Если не совпадают, рисовать с запасом.
- `configs/scene.json` **закоммитить**.

## 6. Review-видео и быстрая dev-разметка (2–3 часа на троих)

```bash
python tools/run_rules.py --cache cache --out dev_pred.json
python tools/render_review.py DataSets/C3896.MP4 --cache cache --out review
```

- В `review/C3896.MP4_review.mp4` видны: боксы с ID и скоростью (BS/с), зелёным — выученная маска дороги, жёлтым — переходы, сверху — активные события, снизу — шкала событий.
- Смотреть на 2× в VLC или mpv. Для каждого события **по конвенциям организаторов** (таблица ниже) записать строку в `dev_labels/labels.csv`:
  ```csv
  video,start,end,label
  C3896.MP4,62.4,91.0,stopped_vehicle
  C3896.MP4,3:21.3,3:25.9,jaywalking
  ```
- Размечать **все** события, а не только те, что нашёл конвейер: пропуски нам тоже важны.
- Классы: сначала 5 реализованных. Если увидели другой класс (проезд на красный, near-miss) — тоже записать, это сигнал, что его стоит сделать.
- Распределение: 4 видео на троих. Одно видео размечают двое независимо, потом прогоняем `evaluate.py` (разметка A как GT, разметка B как предсказания), чтобы увидеть, насколько люди сходятся.

```bash
python tools/csv_to_gt.py dev_labels/labels.csv --videos DataSets --out dev_labels/dev_gt.json
```

| Класс | Начало | Конец |
|---|---|---|
| `accident` | первый кадр, где виден контакт | все участники остановились или покинули кадр |
| `near_miss` | начало уклонения (торможение, манёвр) | участники разъехались |
| `red_light` | перёд ТС пересекает стоп-линию на красный | ТС покинуло перекрёсток или кадр |
| `wrong_way` | ТС въезжает на встречную | вернулось в свою полосу или покинуло кадр |
| `stopped_vehicle` | **момент остановки** (не +10 с); стоит ≥ 10 с, не в очереди у светофора | поехало или убрано |
| `jaywalking` | пешеход ступил на проезжую часть вне перехода | ушёл с проезжей части |
| `congestion` | очередь перестала двигаться по всем полосам направления | очередь рассосалась |
| прочие | см. таблицу в `docs/WestCV_plan_RU.pdf`, §5.1 | |

Если событие уходит за конец видео, конец = длительность. Одновременные события одного класса — один сегмент.

## 7. Подбор порогов — наша «тренировка»

```bash
python tools/run_rules.py --cache cache --out dev_pred.json --gt dev_labels/dev_gt.json
```

Печатается официальный F1 по каждому классу и каждому τ (0.3 / 0.5 / 0.7). Меняем `configs/pipeline.yaml` и перезапускаем. Каждый прогон занимает секунды.

| Класс | Много ложных (FP) | Много пропусков (FN) | Промах только при τ = 0.7 |
|---|---|---|---|
| `stopped_vehicle` | ↑ `min_flow_fraction`, ↓ `queue_fraction`, ↓ `stop_speed` | ↓ `min_on_road`, ↑ `stitch_max_gap_s` | проверить `stop_speed`, `stop_window_s` (старт должен быть в момент остановки) |
| `jaywalking` | ↑ `edge_margin_cells`, ↑ `min_duration_s`, дорисовать `sidewalk` | ↓ `edge_margin_cells`, ↓ `road_min_tracks` | ↓ `gap_s` |
| `wrong_way` | ↑ `min_path_bs`, ↑ `flow_min_coherence`, ↑ `min_duration_s` | ↓ `opposite_dot`, ↓ `min_path_bs` | ↓ `min_duration_s` |
| `congestion` | ↑ `min_duration_s`, ↑ `min_vehicles`, ↓ `crawl_speed` | ↓ `min_duration_s`, ↑ `crawl_speed` | ↑ `gap_s` (склейка), старт — момент остановки очереди |
| `accident` | ↑ `pre_speed`, ↓ `max_decel_s`, ↑ `contact_iou` | в сэмплах, скорее всего, нет аварий — не ослаблять вслепую | — |

Правила принятия решений:
- **Класс с ложными срабатываниями на сэмплах, которого нет в dev-разметке, — выключить** (`enabled: false`). Одно ложное срабатывание несуществующего класса добавляет в среднее ноль.
- Разницу меньше ~0.1 F1 считать шумом: событий мало.
- Меняем 2–4 параметра на класс и берём середину плато, а не острый максимум.
- `accident` оставляем включённым: аварии в тесте вероятны, класс и так попадёт в среднее.

## 8. Время и бюджет

- Part A сама считает свой бюджет: читает 45 кадров как харнесс, получает `h` и берёт `T_A = (3 − 1.25·h − 0.25) × длительность`. При исчерпании бюджета проход останавливается, и события строятся по увиденному (в stderr: `time budget reached`).
- Part B сам выключается, если не успевает (`Part B: switched off`), и без CUDA не работает вовсе.
- Если цикл харнесса настолько медленный, что на Part A остаётся меньше 1× длительности, Part B для этого видео выключается заранее (в логе `(Part B off)`), и его время отдаётся Part A.

Если на T4 `total_sec / duration` выше 2.5, меняем по порядку:
1. `decode.ffmpeg_input_args: ["-skip_loop_filter", "all"]` — декод примерно на 10% быстрее.
2. `decode.analysis_fps: 7.5`, `risk.fps: 7.5`.
3. `decode.width: 960`, `detector.imgsz: 960`.
4. `risk.enabled: false` — крайняя мера. Part B стоит всего 0.02–0.05 отбора, пустое видео стоит гораздо больше.

## 9. Внешние датасеты — только если останется время

Для v1 не нужны. Если в воскресенье утром есть свободный человек с GPU (Kaggle T4):

| Зачем | Датасет | Как | Лицензия |
|---|---|---|---|
| Проверить правило `accident` на реальных авариях | ACCIDENT (Kaggle `picekl/accident`) | Прогнать `cache_perception.py` + `run_rules.py` на 20–30 клипах с неподвижных камер и посмотреть, ловится ли авария и нет ли ложных срабатываний | NC-SA: **только проверка**, не обучать на нём Ultralytics-модель |
| Детектор огня и дыма | D-Fire (GitHub `gaia-solutions-on-demand/DFireDataset`, ~3 ГБ) | `yolo train model=weights/yolo26n.pt data=dfire.yaml imgsz=640 epochs=50 seed=0 deterministic=True` на Kaggle T4, ~1–2 ч; скрипт обучения положить в `training/` | CC0, можно |

Всё, что использовали для обучения или проверки, записать в README (датасет и лицензия). Класс `fire_smoke` включать, только если на всех 4 сэмплах **ноль** ложных срабатываний.

## 10. Прогон на Kaggle T4 (обязательно до сдачи) — одной командой, всё приватно

`kaggle/push.sh` загружает код (только закоммиченные файлы) **приватным** датасетом `antigradient-code` и запускает **приватное** GPU-ядро `antigradient-samples-t4`. Ядро выполняет `kaggle/run_samples.py`:
установка → тесты → замер времени → официальный харнесс на всех сэмплах → `--validate-only` → кэш детекций → review-видео.

```bash
pip install kaggle                          # токен: ~/.kaggle/access_token (Windows: %USERPROFILE%\.kaggle\access_token)
KAGGLE_USER=<ваш_логин> bash kaggle/push.sh                          # видео ядро скачает само по ссылкам организаторов
KAGGLE_USER=<ваш_логин> SAMPLES_DIR=DataSets bash kaggle/push.sh     # если Drive отдаёт «Quota exceeded»: загрузить свои (~20 ГБ, долго)
kaggle kernels status <ваш_логин>/antigradient-samples-t4
kaggle kernels output <ваш_логин>/antigradient-samples-t4 -p kaggle_out
```

- На Windows запускать в Git Bash.
- Для интернета в ядре аккаунт Kaggle должен быть подтверждён по телефону.
- В `kaggle_out/` будут:
  - `summary.json` — время каждого шага и **`x_duration` = total_sec / duration для каждого видео** (цель ≤ 2.5);
  - `predictions_samples.json`;
  - `cache/*.npz` — положить в `cache/` для подбора порогов (§7);
  - `review/*_review.mp4` — review-видео для разметки (§6) и для сайта.
- **Финальный `predictions_samples.json` брать из прогона финального коммита.**
- Ничего не публиковать: датасеты создаются без `--public`, ядро с `is_private: true`.

## 11. Сайт и live-демо (25% отбора)

Рубрика сайта: live-демо 30%, визуализации сэмплов 20%, EDA 15%, подход и отчёт 15%, команда 10%, дизайн 10%.
Всё уже свёрстано в `site/` (главная `index.html` + `report.html`). Данные для графиков и картинок собираются
одной командой из результатов прогона на T4; сам сайт статический (GitHub Pages).

**1. Данные сайта** (после §10, из финального прогона):
```bash
python tools/build_site_data.py --cache kaggle_out/cache --pred kaggle_out/predictions_samples.json \
    --gt dev_labels/dev_gt.json --videos DataSets [--video-urls site/videos.json]
```
- Пишет `site/data/site.json`, тепловые карты, поле направлений и по кадру на каждое событие (с рамками треков).
- `--videos` нужен для кадров событий и таблицы кодеков; `--gt` — для сравнения с разметкой и таблицы F1.
- `site/data/` коммитим (≈10–20 МБ). Локально посмотреть: `python -m http.server -d site 8000` → http://localhost:8000
  (открывать через сервер, а не двойным кликом: `fetch` из `file://` не работает).

**2. Команда и ссылки** — `site/config.json`:
- `team`: имя, роль, фото (`assets/team/<имя>.jpg`, квадрат ~400 px), GitHub/LinkedIn, 2–4 пункта «что сделал».
  Пустые записи не показываются. Это 10% сайта за полчаса работы.
- `space_url` — страница Space (`https://huggingface.co/spaces/<user>/antigradient-demo`),
  `space_embed` — сам апп (`https://<user>-antigradient-demo.hf.space`), он встраивается в страницу.
- Тот же состав команды вписать в таблицу Team в `README.md`.

**3. Демо на Hugging Face Space** (Gradio 6.28.0, CPU):
```bash
bash tools/build_space.sh                               # собирает dist/space: app.py, src/, configs/, yolo26n.pt
pip install -U huggingface_hub && hf auth login
hf repos create <user>/antigradient-demo --type space --space-sdk gradio --public   # CLI huggingface_hub 2.x
hf upload <user>/antigradient-demo dist/space . --repo-type space
```
- С июля 2026 новый Gradio Space требует PRO (или платное железо CPU Upgrade, $0.03/ч). Бесплатный CPU basic
  засыпает после простоя — перед проверкой открыть Space, чтобы он проснулся.
- Проверить: загрузить 30-секундный кусок сэмпла (`ffmpeg -i C3896.MP4 -t 30 -map 0:v:0 -c copy clip.mp4`).
- Локально то же самое: `python app/app.py` → http://127.0.0.1:7860.

**4. (Необязательно) review-видео на сайте.** Если выложить `review/*_review.mp4` в HF dataset
(`hf repos create <user>/antigradient-media --type dataset --public`, затем `hf upload <user>/antigradient-media review . --repo-type dataset`) и записать в `site/videos.json`
`{"C3896.MP4": "https://huggingface.co/datasets/<user>/antigradient-media/resolve/main/C3896.MP4_review.mp4", ...}`,
то на сайте вместо кадров будет плеер, а клик по событию перематывает видео. Без этого работают кадры событий.

**5. Публикация.** Settings → Pages → Source: **GitHub Actions**. Workflow `.github/workflows/pages.yml` выкладывает
`site/` при каждом пуше в `main`, затрагивающем `site/`. Адрес: `https://<owner>.github.io/WestCV/`.
- Pages для **приватного** репозитория доступен только на платном плане GitHub; к сдаче репозиторий и так должен быть открыт.
- Сайт, Space и репозиторий — публичные по условиям задания. Приватными остаются Kaggle-ноутбуки и датасеты (§10).
  Space содержит только код и `yolo26n.pt`, никаких видео.

## 12. Сдача (до 21:00 воскресенья)

- [ ] `pytest` зелёный, `evaluate.py --validate-only` → VALID.
- [ ] `predictions_samples.json` в корне, получен финальным кодом на T4.
- [ ] README: команда, кто что сделал, ссылка на сайт. `site/config.json` заполнен, `site/data/` из финального прогона.
- [ ] `research/` убрать или вынести из репозитория (рубрика «no dead code»).
- [ ] `git tag v1.0 && git push origin v1.0`. После тега ничего не менять.
- [ ] Ссылка на репозиторий (тег) и на сайт отправлена в форме хакатона.

## 13. Если что-то сломалось

| Симптом | Что делать |
|---|---|
| `weights not found` | `bash weights/download.sh` |
| `Part B disabled: no CUDA device` | Нормально на машине без GPU. На T4 такого быть не должно — проверить `torch.cuda.is_available()` |
| `ffmpeg produced no frames` | `decode.backend: cv2` в конфиге |
| `time budget reached` | §8 |
| 0 событий на всех видео | Посмотреть review-видео: есть ли треки, зелёная ли дорога. Проверить `scene.json` и `road_min_tracks` |
| Много `wrong_way` на перекрёстке | ↑ `flow_min_coherence`, ↑ `min_path_bs` |
