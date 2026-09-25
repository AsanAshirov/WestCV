# Инструменты, модели и датасеты для проекта WIUT CV Track

Каталог собран 2026-09-25 для команды из 3 человек (WIUT Hackathon 2026, трек Computer Vision, отборочное задание).

- **Как проверяли.** Версии, лицензии и размеры сверены 2026-09-25 по PyPI JSON, файлам LICENSE на GitHub, API Hugging Face, HTTP HEAD на URL весов, API Kaggle, arXiv и Zenodo.
- **Непроверенное.** Метка **«(не подтверждено)»** означает, что это поле проверить не удалось.
- **Скорости на T4.** Если цифра не взята из таблицы авторов модели, это оценка, и она так и помечена.
- **Рекомендации.** **основной** = ставим/используем; запасной = план Б; опционально = только если останется время или появится пробел; избегать = не брать (лицензия, облако, устарело, не подходит).
- **Исключено.** Из каталога убраны источники, данные которых больше нельзя скачать или получить за время хакатона (мёртвые ссылки Google Drive, только протухшие YouTube-ссылки, доступ только по письму автору).

> **Сроки и лимиты времени («сколько осталось по времени?»).**
> - **Дедлайн.** В материалах задания (PDF задания, `Videos.pdf`, стартовый kit `wiut_cv_scripts.zip`) дата и время дедлайна **не указаны**. Там есть только фразы «The tagged commit at the deadline is what we run» и что железо «may update … before the deadline». Поэтому посчитать остаток времени нельзя: дедлайн нужно уточнить у организаторов (канал/чат хакатона).
> - **Сейчас** по часам ноутбука: пятница, 2026-09-25, ~19:05 (UTC+5). Файлы задания сохранены в папку проекта сегодня в 14:06–14:13.
> - **Трудозатраты на разметку** (оценка, не замер): ~6–8 ч wall-clock на ~25 мин сэмплов при 3 людях (§1.2).
> - **Лимит на обработку.** Жёсткий лимит в документах один: на каждое видео **Part A + Part B вместе ≤ 3 × длительность видео** по wall-clock (`TIME_FACTOR_DEFAULT = 3.0` в `run_submission.py`). Если видео не укладывается, оно засчитывается пустым, включая события Part A.

---

## 0. Коротко: что делаем сегодня

1. **Одним сообщением спросить организаторов:**
   - дедлайн (дату и час);
   - файл `camera.md`: PDF разрешает хардкодить из него факты сцены, но в `wiut_cv_scripts.zip` его нет;
   - формат скрытых тестовых видео: PDF пишет «typical clips are several minutes long at 25 fps», а сэмплы — 29.97p 4K H.264 4:2:2 10-bit;
   - где начинается `stopped_vehicle`: в момент остановки или через 10 с после неё.
2. **Сделать прокси-видео** командой ffmpeg из §1.3: 1080p, H.264 4:2:0 8-bit, **то же число кадров**, наложенный номер кадра. Каждое проверить через ffprobe: число кадров, `r_frame_rate`, `start_time`, совпадение длительностей аудио и видео.
3. **Разметка событий: Label Studio 1.23.0** (Apache-2.0).
   - Поднять на одном ноутбуке через `uv venv --python 3.12 lsenv` в **cmd**. `py -3.12` на этом ноутбуке не работает, а `&&` не работает в PowerShell 5.1.
   - Завести отдельный проект на каждого аннотатора.
   - Конфиг, гайдлайн и конвертер в `ground_truth.json` — в §1.2–1.4.
4. **Геометрия сцены: labelme 7.7.0** на полноразмерном PNG (`-pix_fmt rgb24`) из **оригинального** 4K-файла, результат — `scene.json`.
5. **Боксы: X-AnyLabeling 4.0.6.** Кадры извлекаем из оригиналов, предразметку делаем YOLO и open-vocabulary промптами, экспорт — YOLO/COCO.
6. **Детектор и трекер.**
   - `ultralytics==8.4.163`, YOLO26m (s, если не хватает времени), `imgsz=1280`, FP16.
   - Трекер ByteTrack или TrackTrack с `gmc_method: none` и `track_buffer: 90`.
   - Декодирование через PyAV (`av`) с уменьшением кадра в swscale.
   - Пакет OpenCV ровно один: `opencv-python==4.14.0.94`, без `-headless`.
   - ⚠️ *Примечание при сведении документов:* отчёт `WIUT_CV_Track_analysis_RU.md` (§3.2, §11.1) пинит `opencv-python-headless==4.13.0.92`, и это подкреплено замером скорости `cv2.read()` харнесса под Linux. Пока пин не выбран замером полного `run_submission.py` на реальных сэмплах, опираемся на отчёт. Два пакета OpenCV в одном окружении не ставим.
7. **Сегодня же замерить полное время `run_submission.py` на сэмплах.**
   - Part B харнесса читает **каждый** 4K-кадр через `cv2.VideoCapture.read()`. На ноутбуке на синтетике того же формата это 14–23 fps, то есть ≈1.3–2.2× длительности уходит только на Part B.
   - На весь Part A остаётся ≈0.8–1.7×, подробности в §2.3.6.
8. **Датасеты скачивать в таком порядке** (лучше прикреплять в Kaggle-ноутбуках, а не качать локально):
   - D-Fire (CC0, ~3.1 GB);
   - ACCIDENT (`picekl/accident`, 20.16 GB);
   - MIO-TCD Localization (3.74 GB) и/или UA-DETRAC;
   - TAD (зеркало на Kaggle, 13.44 GB).

   Параллельно отправить заявку на SinD (по e-mail) и AI City Challenge Datasets Request Form.
9. **Модель событий: VideoMAE V2 distilled ViT-S/16** (44.3 MB, код MIT, HF-репо `apache-2.0`). Дообучать на Kaggle T4 **после** того, как заработают правила. VLM-верификатор по умолчанию **не берём (no-go)**, включаем только по критериям из §2.3.4.
10. **Лицензии.**
    - С Ultralytics весь публичный репозиторий распространяется под **AGPL-3.0**.
    - Не брать веса с условиями non-commercial или no-redistribution: DEIMv2, FireViewer, RF-DETR-XL/2XL (PML 1.0), SAM 3, модели на CrowdHuman.
    - Каждый датасет и его лицензию указать в README (§3.3).
    - До обучения детектора решить развилку «AGPL-веса Ultralytics против NC-SA-датасетов» (UA-DETRAC, MIO-TCD, ACCIDENT) — варианты в §3.3.

---

## 1. Инструменты разметки видео

Задач разметки три:

- **(1) временные сегменты событий:** 14 классов, которые могут перекрываться, с точностью до кадра;
- **(2) боксы и треки** для дообучения детектора;
- **(3) геометрия сцены:** полосы, стоп-линии, переходы, сплошные линии, ROI светофоров, зоны запретов.

**Про формат видео.** Браузеры, Windows Media Foundation и JavaFX, как правило, не воспроизводят H.264 High 4:2:2 10-bit, а NVDEC на Turing (T4, GTX 1650) его не принимает. Поэтому всем браузерным и десктопным плеерам нужен **прокси** (§1.3). Декодеры на базе FFmpeg читают оригинал напрямую: CVAT-сервер через PyAV, OpenCV 5.0.0, mpv. Это проверено на синтетическом клипе 3840x2160 `yuv422p10le` 30000/1001.

### 1.1 Сводная таблица

| Инструмент (версия) | Для чего | 4K 4:2:2 10-bit напрямую? | Экспорт | Установка | Лицензия | Рекомендация |
|---|---|---|---|---|---|---|
| **Label Studio Community 1.23.0** (Video + TimelineLabels) | Задача 1 (основной); задача 2: RectangleLabels на кадрах, VideoRectangle-треки | Нет (браузер) → прокси H.264 yuv420p CFR | JSON: кадровые диапазоны, 1-based, включительно | `uv venv --python 3.12` + `uv pip install label-studio` (Python >=3.10,<4) или Docker `heartexlabs/label-studio:latest` (~371 MB); 20–40 мин | Apache-2.0 (Enterprise — коммерческая) | **основной** |
| **X-AnyLabeling 4.0.6** | Задача 2: боксы/треки с предразметкой; задача 3 возможна (полигоны, линии) | Кадры извлекаем ffmpeg из оригинала. Плеер Video Classifier (Qt6 QMediaPlayer) с 4:2:2 — (не подтверждено) | YOLO / COCO / MOT, labelme-JSON | `uv pip install "x-anylabeling-cvhub[gpu]"` (CUDA 12 + cuDNN 9 DLL) или `[cpu]`; Python >=3.11; 15–30 мин + загрузка моделей | GPL-3.0 (только сам инструмент; разметка ваша; у скачанных весов свои лицензии, напр. YOLO — AGPL-3.0) | **основной** |
| **labelme 7.7.0** | Задача 3: геометрия на опорном кадре | Да: работаем с PNG rgb24, извлечённым из оригинала | labelme-JSON → свой `scene.json` | `pip install labelme` (requires-python >=3.12; колёса есть и для 3.14); 5 мин + 30–60 мин рисования | GPL-3.0 (JSON ваш) | **основной** |
| **FFmpeg** (локально N-123094-g561f37c023-20260301) | Прокси, кадры, опорные PNG — для всех задач | Да | — | Уже установлен | Локальная сборка — GPLv3 (`--enable-gpl --enable-version3`); используем как внешний инструмент, в репо не кладём | **основной** |
| **uv 0.10.6** (последняя 0.12.19) | Python 3.12 venv для LS / X-AnyLabeling / FiftyOne / Datumaro | — | — | Уже установлен | MIT OR Apache-2.0 | **основной** |
| **CVAT Community v2.76.0** (self-hosted, Docker) | Задача 2: треки с интерполяцией, авторазметка `cvat-cli`; альтернатива для задачи 3. Для задачи 1 **не подходит** | Да: декодирует сервер (PyAV 18.1 читает `yuv422p10le`; 3840x2160 = 8 294 400 px < лимита OpenH264 9 437 184, issue #7425) | COCO / YOLO / MOT / CVAT XML | WSL2 + Docker Desktop + Chrome; `git clone --branch v2.76.0`; 1–2 ч | MIT; функции и веса на Ultralytics — AGPL-3.0 | запасной |
| **ELAN 7.1** | Задача 1, десктопный бэкап: отдельный тир на класс | Нет → прокси (`-g 15 -bf 0`) | `.eaf`, tab-delimited; парсинг через `pympi-ling` 1.71 | `ELAN_7-1_win.exe` (85.7 MB; есть .msi и .zip); 20–30 мин | GPL-3.0 (+ условия использования и просьба цитировать) | запасной |
| **BORIS 9.15.0** | Задача 1: быстрое клавиатурное кодирование (state events) | Плеер mpv (декодеры FFmpeg); скорость 4K 4:2:2 на ноутбуке (не подтверждено) → лучше прокси | TSV / CSV / XLSX | **Windows Portable** `boris-9.15.0-win64.zip` (master теперь требует ровно Python 3.13); 30 мин | GPL-3.0 | опционально |
| **VIA 3.0.13** (VGG Image Annotator) | Задача 1 без установки; фолбэк для задачи 3 | Нет (HTML5) → прокси | CSV / VIA3 JSON (секунды) | `via-3.0.13.zip` (8 636 593 байт) → `via_video_annotator.html`; 5 мин | BSD-2-Clause | опционально |
| **Свой маркер сегментов** (OpenCV/PyAV + PySide6) | Задача 1; просмотр GT против предсказаний | Да: OpenCV 5.0.0 и PyAV 18.1.0 декодировали 120/120 кадров синтетики | Свой формат (индексы кадров) | `pip install opencv-python numpy av`; MVP ~3–4 ч (оценка) | Своя; opencv-python — Apache-2.0, PyAV — BSD-3-Clause (в колёсах FFmpeg под LGPL/GPL) | опционально |
| **FiftyOne 1.22.0** | Ревью GT против предсказаний, быстрая tIoU-проверка (ActivityNet mAP) | Нет (браузер) → прокси | `fo.TemporalDetection` | `uv pip install fiftyone`, `FIFTYONE_DO_NOT_TRACK=true`; 30–60 мин на загрузчик (оценка) | Apache-2.0 | опционально |
| **mpv v0.41.0** | Покадровый просмотр оригинала, чтобы решать спорные границы | Да (FFmpeg); real-time на i5-12450H (не подтверждено) | — (только просмотр) | Сборки shinchiro / zhongfly (ссылки с mpv.io) | GPLv2+ (LGPLv2.1+ при `-Dgpl=false`) | опционально |
| **Datumaro 1.13.11** | Слияние и конвертация экспортов боксов | — | coco, cvat, yolo, labelme, mot, voc, roboflow, video | venv 3.12: колёса только cp310–cp313 | MIT | опционально |
| **makesense.ai 1.11.0-alpha** | Экстренный фолбэк для геометрии и простых боксов | Только изображения | VGG JSON/COCO (полигоны), CSV (линии) (не подтверждено) | Сайт в браузере | GPL-3.0 | опционально |
| **DarkLabel 2.4** | Быстрые MOT-боксы на прокси (Windows) | (не подтверждено) → прокси | VOC, darknet YOLO, XML/TXT (через `darklabel.yml`) | `DarkLabel2.4.zip` из GitHub release | Закрытый freeware, только некоммерческое использование | опционально |
| **CVAT Online** (app.cvat.ai) | Хостинг CVAT | Сервер | Как у CVAT | Регистрация | SaaS (ядро MIT) | избегать: на free-плане 1 GB и нет командной работы; футаж организаторов уходит в облако |
| **Supervisely Community** | Хостинг видео-разметки | — | — | Регистрация | SaaS; SDK — Apache-2.0 | избегать: максимум 2 участника, 5 GB, архивация через 30 дней неактивности |
| **Roboflow Annotate** | Хостинг разметки изображений | — | — | — | SaaS | избегать: Public plan публикует данные на Roboflow Universe, только 2 пользователя |
| **Encord / V7 Darwin / Dataloop** | Корпоративные платформы | — | — | — | Проприетарный SaaS | избегать |
| **Xtreme1 v0.9.2** | Изображения / LiDAR | — | — | Docker | Apache-2.0 | избегать: нет видео-таймлайна |
| **ANVIL 6.0** | Временная разметка (исследовательский инструмент) | Нет (JavaFX, H.264 8-bit) | — | Только по e-mail `download@anvil-software.de` | Бесплатно для research/education | избегать |
| **Kinovea 2025.2.0** | Спорт-анализ, замеры | (не подтверждено) | Нет мультиклассового экспорта интервалов | — | GPL-2.0 | избегать |
| **AnyLabeling v0.4.43** | Десктопная разметка изображений | — | — | — | GPL-3.0 | избегать: вытеснен X-AnyLabeling |

#### Важные детали по основным инструментам

**Label Studio (задача 1).**

*Как устроены кадры.*
- `TimelineRegion` хранит целые `start`/`end`. В панели региона есть редактируемые поля **«Start frame» / «End frame»**.
- Диапазоны **1-based и включительные**. Официальный конвертер в `label_studio_ml/examples/yolo/utils/converter.py` (именно этот путь, не `utils/converter.py`) переводит их в `start-1..end`. Поэтому `start_sec = (start-1)/fps`, `end_sec = end/fps`.
- Регионы разных классов могут перекрываться.
- Перемотка в `VideoCanvas.tsx` идёт на `(frame-1)/framerate`, отображаемый кадр — `ceil`/`round(currentTime*framerate)`, минимум 1. Дрейфа float нет.

*Что обязательно настроить.*
- **`frameRate` по умолчанию 24.** Его надо задать: `frameRate="$fps"` в конфиге и `fps` в данных задачи. Иначе съезжают все метки.
- **Минимальная скорость воспроизведения.** В исходниках (`Video.js`) нижняя граница по умолчанию 0.25, в документации написано 1. Надёжнее явно указать `minPlaybackSpeed="0.25"`, тогда 0.5x работает.
- **Формат видео по документации:** MP4 / H.264 / AAC, `yuv420p`, постоянный fps. Там же предупреждение: «All audio and video streams from your file must also have the same durations; otherwise, you will have extra total frames». **Команду из документации с `-r 30` не копировать:** при 29.97 она меняет число кадров.

*Ограничения Community-версии.*
- Нет матрицы согласия между аннотаторами.
- Видят ли аннотаторы вкладки друг друга (не подтверждено). Безопасный вариант — **отдельный проект на аннотатора**.
- Производительность таймлайна примерно на 10k кадров (не подтверждено).

*Python.*
- На 3.14 зависимости резолвятся (только чистый Python-пакет `attr` идёт как sdist), но сам запуск на 3.14 (не подтверждено).
- `py -3.12` на этом ноутбуке **падает** («No suitable Python runtime found»). Используем `uv venv --python 3.12` (проверено: создаёт venv с 3.12.12) или `py -V:Astral/CPython3.12.12 -m venv lsenv`.

*Что ещё умеет.* ML-бэкенды (`label-studio-ml-backend`) для YOLO, Grounding DINO и SAM2-video. Выход модели можно импортировать как `predictions` и проверять.

**X-AnyLabeling (задача 2).**
- **Video Classifier для задачи 1 не подходит.** В `video_classifier/timeline.py` функция `_clamp_created_range` ограничивает новый сегмент соседними, поэтому сегменты не могут перекрываться.
- **Чат/VLM-модуль.** По умолчанию он использует локальный провайдер `ollama`, но в списке есть и облачные (Anthropic, DeepSeek, Gemini, OpenAI и др.). С футажем организаторов — **только локальный Ollama**. Вызывает ли «AI auto segmentation» в Video Classifier облачные API (не подтверждено).
- **Экстра `[gpu]`.** Ставит `onnxruntime-gpu >=1.18.1,<1.27` под CUDA 12.x, для него нужны DLL CUDA 12 и cuDNN 9 (на GTX 1650 — не подтверждено). `[gpu-cu13]` — для CUDA 13.
- **Отдельный venv.** Не ставить в один venv с `opencv-python`: X-AnyLabeling тянет `opencv-contrib-python-headless`.
- **Один пользователь.** Папки кадров делим между людьми, экспорты сливаем (Datumaro).

**labelme (задача 3).**
- Типы фигур: polygon, rectangle, circle, line, linestrip, point; у фигур есть `group_id`, `description`, `flags`.
- Есть точка → полигон в стиле SAM через osam/onnxruntime.
- CLI v7: `--labels`, `--validate-label`, `--output` (ожидает каталог), `--with-image-data` (проверено в `__main__.py`).
- Схемы атрибутов нет, поэтому команде нужно строго держаться соглашения об именах (§1.4).

**CVAT (запасной для задачи 2).**
- Интервальные метки в 2026 есть **только для аудио** (тип `interval` в CHANGELOG).
- SAM2-трекер в Nuclio-варианте «Available only for Enterprise deployments», в AI-agent-варианте — «Available for CVAT Online and Enterprise».
- Ветка по умолчанию `develop` тянет `cvat/server:dev`. Поэтому клонировать **`--branch v2.76.0`**: compose этой версии по умолчанию берёт тег `v2.76.0`.
- Образы: `cvat/server:v2.76.0` ~701 MB, `cvat/ui` ~44 MB. Плюс postgres, redis, kvrocks, clickhouse, grafana, opa, vector, traefik. Поддерживается только браузер Chrome.

**ELAN (запасной для задачи 1).**
- 14 тиров из шаблона `.etf` с контролируемым словарём, поэтому перекрытия естественны.
- Времена хранятся в мс и к кадрам не привязаны: снапать к кадрам в конвертере.
- Покадровый шаг и NTSC-отображение кадров описаны в мануале (в этом проходе не подтверждено).
- На Windows использует DirectShow / MMF / JavaFX / VLC, поэтому нужен прокси.

**BORIS.**
- 14 классов задаются как state events: нажатие клавиши — начало, повторное нажатие — конец. Смотреть на 0.5x.
- В исходниках есть модуль `irr.py` (Cohen's kappa).
- Точные границы править менее удобно, чем вводить номер кадра в LS.

**VIA3.**
- Добавляем атрибут `event` с 14 вариантами: на таймлайне по строке на вариант, классы могут перекрываться.
- **Не знает fps:** в исходнике шаг 1/50 с и `toFixed(3)`, точность около ±1 кадр.
- Сервер shared-проектов (VGG) **не использовать**.

**Свой маркер.**
- Окно cv2 на прокси или оригинале: покадровый шаг, `[`/`]` ставят начало/конец, горячие клавиши классов, полоса таймлайна. Сохраняем индексы кадров и переводим через `30000/1001`.
- Обратная перемотка по long-GOP оригиналу медленная и неточная. Используем прокси (`-g 15 -bf 0`) или кольцевой буфер.
- Скорость декодирования 4K зависит от клипа: при перепроверке OpenCV дал 31 fps, PyAV 41 fps; в более ранних тестах было 19.5 и 67 fps.

**FiftyOne.**
- `fo.TemporalDetection`: `support` — это `[first, last]` кадры включительно; есть `TemporalDetection.from_timestamps`.
- `fiftyone/utils/eval/activitynet.py` (`iou_threshs`) — это **mAP, а не macro-F1 организаторов**. Авторитетен только `evaluate.py`.
- Совместимость с Python 3.14 (не подтверждено), используйте 3.12.

**mpv.**
- `.` и `,` — шаг на кадр вперёд/назад.
- Свойства `estimated-frame-number` и `container-fps` можно вывести в OSD.
- Команда: `mpv --osd-level=3 --osd-status-msg="f=${estimated-frame-number} t=${time-pos}" CLIP01.MP4` (имена свойств проверены, точная строка OSD не тестировалась).

### 1.2 Процесс разметки для команды из 3 человек и гайдлайн по границам

#### Выбор инструментов

| Задача | Основной | Запасной |
|---|---|---|
| 1. События: 14 перекрывающихся классов, точность до кадра | **Label Studio 1.23**: Video + TimelineLabels (индексы кадров, перекрытия, многопользовательский) | ELAN 7.1 (тир на класс); BORIS 9.15 (быстрый клавиатурный кодинг) |
| 2. Боксы (и треки) с предразметкой | **X-AnyLabeling 4.0.6** (локально; YOLO / Grounding DINO / YOLOE / SAM) | CVAT v2.76 self-hosted (треки + `cvat-cli task auto-annotate`) |
| 3. Геометрия сцены | **labelme 7.7.0** на 4K PNG | Полилинии/полигоны в CVAT; VIA / makesense |

#### Поднимаем Label Studio (хост — один ноутбук; в **cmd.exe**, не в PowerShell 5.1)

Раскладка папок (пример): `C:\westhack\raw` — оригиналы, `C:\westhack\proxy` — прокси, `C:\westhack\frames` — кадры, `C:\westhack\geometry` — геометрия.

```bat
cd /d C:\westhack
uv venv --python 3.12 lsenv
lsenv\Scripts\activate
uv pip install label-studio
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=C:\\westhack
label-studio start -p 8080
```

- В PowerShell venv активируется так: `.\lsenv\Scripts\Activate.ps1`, переменные задаются так: `$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"`.
- На Windows документация требует **двойные обратные слэши** в `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`.
- Сокомандники открывают `http://<LAN-IP-хоста>:8080`. Возможно, придётся разрешить порт в брандмауэре Windows.
- Подключить хранилище: Settings → Cloud Storage → Add Source Storage → Local files. URL задач имеют вид `/data/local-files/?d=proxy/CLIP01_p1080.mp4`.
- Альтернатива через Docker: `docker run -it -p 8080:8080 -v ${PWD}/mydata:/label-studio/data heartexlabs/label-studio:latest`.

**Labeling config:**

```xml
<View>
  <Video name="video" value="$video" frameRate="$fps" minPlaybackSpeed="0.25" height="620" timelineHeight="320"/>
  <TimelineLabels name="events" toName="video">
    <Label value="accident" hotkey="1"/> <Label value="near_miss" hotkey="2"/>
    <Label value="red_light" hotkey="3"/> <Label value="wrong_way" hotkey="4"/>
    <Label value="illegal_u_turn" hotkey="5"/> <Label value="stopped_vehicle" hotkey="6"/>
    <Label value="jaywalking" hotkey="7"/> <Label value="failure_to_yield" hotkey="8"/>
    <Label value="illegal_turn" hotkey="9"/> <Label value="solid_line_crossing" hotkey="0"/>
    <Label value="stop_line" hotkey="q"/> <Label value="congestion" hotkey="w"/>
    <Label value="road_obstacle" hotkey="e"/> <Label value="fire_smoke" hotkey="r"/>
  </TimelineLabels>
  <TextArea name="notes" toName="video" placeholder="сомнительные события / комментарии"/>
</View>
```

**Задачи для импорта.** `fps` и `duration` берём из ffprobe **оригинала**. `orig` — имя оригинала: ключи ground truth — это имена оригиналов, **не прокси**.

```json
[{"data": {"video": "/data/local-files/?d=proxy/CLIP01_p1080.mp4", "orig": "CLIP01.MP4", "fps": 29.97002997, "duration": 340.307}}]
```

- **`frameRate` = 29.97002997** (30000/1001) для наших сэмплов. Для других файлов берите fps из ffprobe и **никогда не хардкодьте** его (в PDF упоминаются клипы на 25 fps).
- **Точные границы** правим в панели региона, в полях «Start frame» / «End frame».
- **Слепая двойная разметка.** Проекты `events_A`, `events_B`, `events_C` с одинаковыми задачами плюс проект `gold` для арбитража.
- **Экспорт:** Export → **JSON** (не JSON_MIN).

#### Роли

- **A — пайплайн.** Прокси, хост LS, конвертеры, прогоны `evaluate.py`. Тоже размечает.
- **B — владелец гайдлайна и первый арбитр.** Пишет правила и лист с примерами. Тоже размечает.
- **C — геометрия и боксы.** labelme и X-AnyLabeling. Размечает события на 1–2 видео.

#### План по часам (оценка, не замер)

Весь проход по ~25 минутам сэмплов занимает примерно **6–8 ч wall-clock на троих** (оценка).

1. **Ч0–1, параллельно.**
   - A: прокси (на двух ноутбуках) и LS.
   - B: гайдлайн v1 (ниже).
   - C: опорные кадры и геометрия.
2. **Ч1–1.5, калибровка.**
   - Все трое размечают одни и те же 60–90 с одного клипа, каждый в своём проекте.
   - Для каждой пары: конвертировать и прогнать `evaluate.py`.
   - Разобрать каждое расхождение и заморозить гайдлайн v1.1.
3. **Ч1.5–4, двойная разметка.**
   - Каждое видео размечают двое независимо, пары ротируются: A+B, B+C, C+A.
   - Один проход занимает ≈4–6× длительности видео (оценка): смотрим на 0.5x, у границ шагаем по кадрам.
4. **Ч4–5, согласие и арбитраж.**
   - Прогнать `evaluate.py`: аннотатор X как GT, Y как предсказания. Смотреть per-class F1 при tIoU 0.3/0.5/0.7; F1 симметрична, одного направления достаточно.
   - Список расхождений — скриптом из §1.4.
   - Третий человек решает спорное в `gold`: импортирует X как аннотацию, Y как prediction, и правит.
   - Если межаннотаторский F1@0.5 по классу ниже ~0.7 (эвристика), правило для класса неоднозначно. Сначала чиним гайдлайн.
5. **Боксы, параллельно** (C + кто свободен).
   - **Кадры.** ~1 кадр на 1–2 с, плотнее вокруг каждого события, плюс все кадры с препятствиями и дымом. Итого ~300–600 кадров.
   - **Предразметка.** YOLO11/26: машины, пешеходы, двухколёсные. Для редкого — промпты Grounding DINO / YOLOE: «debris», «fallen object», «animal», «smoke», «fire», «traffic light».
   - **Классы:** `car, bus, truck, motorcycle, bicycle, pedestrian, tl_red, tl_yellow, tl_green, tl_off, obstacle, smoke, fire`.
   - **Правка:** ~20–40 с на кадр (оценка). Экспорт в YOLO + COCO. Если нужны треки — CVAT-треки на прокси.
   - **Hold-out.** Одно видео целиком оставить тестовым и **не** брать с него псевдоразметку для обучения.
6. **Проверка геометрии.** Наложить `scene.json` на первый, средний и последний кадр каждого видео, чтобы поймать сдвиг камеры между записями. Файл геометрии заводим на каждую установку камеры.

#### Гайдлайн: общие правила (повторяют логику скорера, у которого границы решают)

- **Кадры и секунды.** Время всегда считаем по **оригиналу**. 0-based номер кадра `f` читаем с оверлея прокси. Номер кадра в LS = `f + 1`.
  - **start** — первый кадр, на котором определяющее условие видно выполненным.
  - **end** — последний кадр, на котором оно ещё выполнено (включительно).
  - Конвертер пишет `start_sec = f_start/fps` и `end_sec = (f_end+1)/fps`.
- **Сколько ошибки по границе терпит скорер.** Если сдвинуть предсказание на δ, получится IoU = (L−δ)/(L+δ). Для IoU ≥ 0.7 нужно δ ≤ 0.18·L:
  - событие 3 с — это ≈16 кадров допуска;
  - событие 1 с — только ≈5 кадров.

  **Единообразие конвенции важнее точности до одного кадра.** Больше всего внимания требуют короткие классы: `near_miss`, `red_light`, `solid_line_crossing`.
- **Одно событие = один участник (или одна взаимодействующая пара) и один класс.** Две машины, проехавшие на один красный, — два события. Тот же участник и класс с разрывом меньше 1 с — одно событие.
- **Ставим все подходящие классы, перекрытия — норма.** Например: `accident` → `stopped_vehicle` (если ≥10 с) → `congestion`.
- **Событие обрезано клипом** — начало 0.0 или конец = длительности.
- **Сомнительные случаи** пишем в `notes` и решаем арбитражем. Если организаторы опубликуют определения или примеры GT, **они важнее всего, что ниже.**

#### Начало и конец по классам (предлагаемая конвенция команды; сверить со спецификацией организаторов)

| Класс | start | end |
|---|---|---|
| accident | первый кадр физического контакта (или потери управления, которая заканчивается ударом) | все участники остановились / обломки улеглись |
| near_miss | начало уклонения (резкое торможение или манёвр) или вход траекторий в конфликт | конфликт разрешён (пути разошлись, скорости нормальные) |
| red_light | перёд ТС пересекает стоп-линию при красном для его направления | ТС покинуло зону конфликта / перекрёсток |
| wrong_way | ТС впервые движется против направления полосы на проезжей части | покинуло кадр или вернулось в правильное направление |
| illegal_u_turn | начало смены курса внутри зоны запрета разворота | ТС устойчиво едет в обратном направлении |
| illegal_turn | ТС начинает запрещённую траекторию поворота (пересекает вход поворота) | поворот завершён, ТС в полосе выезда |
| solid_line_crossing | первое колесо касается/пересекает сплошную | ТС полностью в другой полосе |
| stopped_vehicle | момент остановки на проезжей части (не в очереди у светофора); событие оставляем, только если стоянка ≥10 с. **Уточнить у организаторов: start — остановка или +10 с** | ТС поехало / покинуло кадр |
| stop_line | ТС остановилось за стоп-линией при красном | тронулось или загорелся зелёный |
| jaywalking | пешеход ступил на проезжую часть вне перехода / против сигнала | пешеход покинул проезжую часть |
| failure_to_yield | ТС въезжает на переход, когда на нём пешеход | ТС покинуло переход |
| congestion | очередь/медленный поток заполняет `queue_zone` (например, скорость около нуля во всех полосах дольше ~10 с) | поток восстановился |
| road_obstacle | объект или животное впервые на проезжей части | убран / ушёл / конец клипа |
| fire_smoke | первый видимый дым или пламя | больше не видно / конец клипа |

### 1.3 Команды ffmpeg: прокси с сохранением числа кадров и таймкодов

**Что проверено.** Команду для Bash прогнали целиком (вместе с drawtext и `consola.ttf`) на синтетическом клипе 3840x2160 H.264 High 4:2:2 `yuv422p10le` 30000/1001 с дорожкой AAC:

- видеопакетов на входе и выходе 120 и 120;
- `start_time` 0, длительность видеопотока 4.004 с в обоих файлах;
- выход High/`yuv420p`;
- аудио (3.994 с) скопировано с той же длительностью.

**Скорость.** Заявлено ~55 fps на клипе 147 Mbps на i5-12450H, только CPU (не перезамерялось). В проверочном тесте 120 кадров 4K перекодировались за ~5.3 с wall-clock. Ориентир: клип 6 мин — это ~3–6 мин перекодирования, прокси ~250–450 MB (оценка).

**Шаг 0. Сравнить длительности аудио и видео в каждом Sony-файле.** Label Studio предупреждает, что разные длительности дают лишние кадры.

```bash
ffprobe -v error -show_entries stream=index,codec_type,duration -show_entries format=duration -of compact=p=0 CLIP01.MP4
```

Если длительность аудио отличается от видео, в команде ниже замените `-map "0:a:0?" ... -c:a aac -b:a 128k` на **`-an`**, то есть просто уберите звук. `-shortest` безопасен, только когда аудио **длиннее** видео. Если аудио короче, `-shortest` обрежет видео и число кадров изменится. После любого варианта перепроверьте число кадров.

**Шаг 1. Один файл (Git Bash), проверенная команда:**

```bash
ffmpeg -hide_banner -y -i CLIP01.MP4 -map 0:v:0 -map "0:a:0?" \
  -vf "scale=1920:-2:flags=bicubic,format=yuv420p,drawtext=fontfile='C\:/Windows/Fonts/consola.ttf':text='f=%{frame_num} t=%{pts\:flt}':x=20:y=20:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.6" \
  -fps_mode passthrough -c:v libx264 -preset veryfast -crf 20 -profile:v high -g 15 -bf 0 \
  -c:a aac -b:a 128k -movflags +faststart proxy/CLIP01_p1080.mp4
```

**Вся папка (PowerShell).** По словам исследователя, та же строка аргументов тестировалась в PowerShell.

```powershell
$src="C:\westhack\raw"; $dst="C:\westhack\proxy"; New-Item -ItemType Directory -Force $dst | Out-Null
Get-ChildItem "$src\*.MP4" | ForEach-Object {
  $out = Join-Path $dst ($_.BaseName + "_p1080.mp4")
  ffmpeg -hide_banner -y -i $_.FullName -map 0:v:0 -map "0:a:0?" -vf "scale=1920:-2:flags=bicubic,format=yuv420p,drawtext=fontfile='C\:/Windows/Fonts/consola.ttf':text='f=%{frame_num} t=%{pts\:flt}':x=20:y=20:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.6" -fps_mode passthrough -c:v libx264 -preset veryfast -crf 20 -profile:v high -g 15 -bf 0 -c:a aac -b:a 128k -movflags +faststart $out
}
```

**Что делает каждая часть:**

- **`-fps_mode passthrough`** — каждый кадр сохраняется с исходным pts, без дублей и пропусков. Заменяет старый `-vsync 0`.
- **Никогда не добавлять** `-r`, фильтр `fps=` или `-ss` перед `-i`.
- **`-map 0:v:0 -map "0:a:0?"`** — отбрасывает потоки таймкода и метаданных Sony. Звук полезен аннотаторам: по нему слышно удар.
- **`format=yuv420p` + `-profile:v high`** — 10-bit 4:2:2 превращается в 8-bit 4:2:0. Это играют браузеры, WMF и JavaFX.
- **`-g 15 -bf 0`** — ключевой кадр каждые 0.5 с и без B-кадров, поэтому перемотка и покадровый шаг быстрые и точные. MPI для ELAN тоже советует B-frames 0.
- **`-movflags +faststart`** — видео сразу стартует и перематывается по HTTP.
- **drawtext** рисует **0-based номер кадра исходника** и pts в секундах; аннотаторы читают границы прямо с оверлея. Ловушка: `%{pts:flt:3}` добавляет *смещение* 3 с, а не задаёт точность. Правильно `%{pts\:flt}` или `%{pts\:hms}`.
- **Лёгкий вариант 720p:** `scale=1280:-2` и `-crf 21`.
- **Чистый прокси** (для визуального ревью или VLM) — без фильтра drawtext.

**Шаг 2. Проверить каждый прокси.** `-count_packets` читает только контейнер, поэтому быстро работает даже на файлах в 6 GB.

```bash
ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets,r_frame_rate,start_time -show_entries format=duration -of csv=p=0 CLIP01.MP4
ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets,r_frame_rate,start_time -show_entries format=duration -of csv=p=0 proxy/CLIP01_p1080.mp4
```

`nb_read_packets`, `r_frame_rate` (30000/1001) и `start_time` должны совпасть. Пакетная проверка для Git Bash (скрипт не тестировался):

```bash
cd /c/westhack
for f in raw/*.MP4; do
  b=$(basename "$f" .MP4)
  q='-v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets,r_frame_rate,start_time -of csv=p=0'
  a=$(ffprobe $q "$f"); p=$(ffprobe $q "proxy/${b}_p1080.mp4")
  [ "$a" = "$p" ] && echo "OK   $b $a" || echo "FAIL $b orig=$a proxy=$p"
done
```

**Длительность `format duration` оригинала** записываем в данные задачи (`duration`) и в `ground_truth.json`.

**Кадры для разметки боксов** извлекаем из ОРИГИНАЛА. Имя файла — индекс кадра исходника (проверено: `f_000270.jpg` — это кадр 270 на 9.009 с).

```bash
# каждый 30-й кадр (~1 кадр/с)
ffmpeg -hide_banner -i CLIP01.MP4 -map 0:v:0 -vf "select='not(mod(n\,30))'" -fps_mode passthrough -frame_pts true -q:v 2 frames/CLIP01_%06d.jpg
# плотнее внутри окна события, например кадры 360..570, каждый 5-й
ffmpeg -hide_banner -i CLIP01.MP4 -map 0:v:0 -vf "select='between(n\,360\,570)*not(mod(n\,5))'" -fps_mode passthrough -frame_pts true -q:v 2 frames/CLIP01_%06d.jpg
```

**Опорный кадр для геометрии:**

```bash
ffmpeg -ss 60 -i CLIP01.MP4 -frames:v 1 -pix_fmt rgb24 ref_CLIP01.png
```

Без `rgb24` из 10-bit исходника может получиться 16-битный PNG.

**Декодирование оригиналов в Python.** Обе библиотеки читают 4:2:2 10-bit (проверено на синтетике):

- `opencv-python` 5.0.0 отдаёт 8-bit BGR;
- PyAV 18.1 отдаёт `yuv422p10le`.

Скорость на 4K зависит от клипа: 19.5–31 fps у OpenCV, 41–67 fps у PyAV в разных тестах.

### 1.4 Конвертер экспорта в `ground_truth.json` (Python)

Целевой формат организаторов:

```json
{"CLIP01.MP4": {"duration": 340.3, "fps": 29.97, "events": [[12.0, 19.0, "accident"]]}}
```

**`ls_to_gt.py` — экспорт Label Studio (JSON) в ground truth.** Диапазоны TimelineLabels 1-based и включительные (`start_sec = (start-1)/fps`, `end_sec = end/fps`). Скрипт прогнан на Python 3.14 на mock-экспорте: фильтр по аннотатору (`completed_by` числом или объектом), выбор последней аннотации задачи, режим `--all`. На реальном экспорте LS 1.23 (не подтверждено) — проверьте на первом же экспорте.

```python
# ls_to_gt.py — Label Studio JSON export -> ground_truth.json
# usage: python ls_to_gt.py export_events_A.json gt_A.json [--user ID_или_email] [--all]
import argparse, json

FPS_DEFAULT = 30000 / 1001

def _user_ok(a, user):
    if user is None:
        return True
    cb = a.get("completed_by")
    if isinstance(cb, dict):                 # в части экспортов completed_by — объект
        return str(user) in (str(cb.get("id")), str(cb.get("email")))
    return str(cb) == str(user)

def ls_to_gt(export_path, out_path, user=None, take_all=False):
    with open(export_path, encoding="utf-8") as fh:
        tasks = json.load(fh)
    gt = {}
    for t in tasks:
        d = t["data"]
        vid = d["orig"]                          # имя ОРИГИНАЛА (ключ GT), не прокси
        fps = float(d.get("fps", FPS_DEFAULT))
        dur = float(d["duration"])
        e = gt.setdefault(vid, {"duration": dur, "fps": round(fps, 2), "events": []})
        anns = [a for a in t.get("annotations", [])
                if not a.get("was_cancelled") and _user_ok(a, user)]
        if anns and not take_all:                # по умолчанию одна (последняя) аннотация на задачу
            anns = [max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")]
        for a in anns:
            for r in a.get("result", []):
                if r.get("type") != "timelinelabels":
                    continue
                labs = r["value"].get("timelinelabels") or []
                for rg in r["value"].get("ranges", []) if labs else []:
                    s = (rg["start"] - 1) / fps  # LS: 1-based, включительно
                    en = min(rg["end"] / fps, dur)
                    e["events"].append([round(s, 3), round(en, 3), labs[0]])
    for v in gt.values():
        v["events"].sort()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(gt, fh, indent=1, ensure_ascii=False)
    return gt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("export"); ap.add_argument("out")
    ap.add_argument("--user", help="id или e-mail аннотатора (поле completed_by)")
    ap.add_argument("--all", action="store_true", help="взять все аннотации задачи, а не последнюю")
    args = ap.parse_args()
    gt = ls_to_gt(args.export, args.out, args.user, args.all)
    print({k: len(v["events"]) for k, v in gt.items()})
```

- Пример: кадры 361..570 в LS (на оверлее `f=360..569`) превращаются в `[12.012, 19.019, "accident"]`.
- По умолчанию берётся **одна** аннотация на задачу (самая свежая по `updated_at`), чтобы случайно не сложить разметку двух людей в один GT. `--all` склеивает все аннотации задачи.
- `fps` в GT округляется до 2 знаков (29.97), как в примере формата организаторов; секунды считаются по точному 30000/1001.

**Проверка согласия.**

1. `python ls_to_gt.py export_events_A.json gt_A.json` и то же самое для B (из проектов `events_A`, `events_B`).
2. Прогнать `evaluate.py` организаторов: `gt_A` как ground truth, `gt_B` как предсказания. CLI подстройте под `evaluate.py`: если у предсказаний другая схема, оберните `events` соответственно.

**`disagree.py` — список несовпавших событий (в обе стороны):**

```python
# usage: python disagree.py gt_A.json gt_B.json [thr=0.5]
import json, sys

def tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0

A = json.load(open(sys.argv[1], encoding="utf-8"))
B = json.load(open(sys.argv[2], encoding="utf-8"))
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
for name, X, Y in (("A->B", A, B), ("B->A", B, A)):
    for vid, v in X.items():
        for ev in v["events"]:
            same = [e2 for e2 in Y.get(vid, {}).get("events", []) if e2[2] == ev[2]]
            best = max((tiou(ev, e2) for e2 in same), default=0.0)
            if best < thr:
                print(name, vid, ev, "best tIoU", round(best, 2))
```

`disagree.py` прогнан на mock-данных (Python 3.14).

**Бэкап на ELAN** (имя тира = класс), `pympi-ling` 1.71 (сниппет не тестировался):

```python
import pympi
FPS = 30000 / 1001
eaf = pympi.Elan.Eaf("CLIP01.eaf")
events = [[s / 1000, e / 1000, tier] for tier in eaf.get_tier_names()
          for (s, e, *_) in eaf.get_annotation_data_for_tier(tier)]
# ELAN хранит мс без привязки к кадрам -> снапаем: round(t*FPS)/FPS
events = [[round(round(s * FPS) / FPS, 3), round(round(e * FPS) / FPS, 3), lab] for s, e, lab in events]
```

**labelme → `scene.json`** (прогнан на mock-JSON labelme, Python 3.14).

- **Метки** (`scene_labels.txt`, по одной в строке): `carriageway`, `lane` (`group_id` = номер полосы), `lane_dir` (linestrip, порядок точек = направление движения), `stop_line`, `crosswalk`, `solid_line`, `signal_roi` (при необходимости отдельные лампы: `lamp_red`, `lamp_amber`, `lamp_green`, `ped_red`, `ped_green`; `group_id` = номер светофорной головы, в `description` — подход), `no_u_turn`, `turn_restriction`, `queue_zone`.
- **Координаты** нормируем по ширине и высоте, тогда правила работают при любом разрешении инференса. Прямоугольник labelme (2 угла) разворачивается в 4 вершины.

```bash
labelme ref_CLIP01.png --labels scene_labels.txt --validate-label exact --output geometry/
```

```python
# usage: python labelme_to_scene.py geometry/ref_CLIP01.json scene_CLIP01.json
import json, sys
from collections import defaultdict

REQUIRED = {"carriageway", "stop_line", "crosswalk", "lane_dir"}   # подстройте под свою сцену

src = json.load(open(sys.argv[1], encoding="utf-8"))
W, H = src["imageWidth"], src["imageHeight"]
shapes = defaultdict(list)
for s in src["shapes"]:
    pts = [[x / W, y / H] for x, y in s["points"]]
    if s.get("shape_type") == "rectangle" and len(pts) == 2:      # 2 угла -> 4 вершины
        (x1, y1), (x2, y2) = pts
        x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    shapes[s["label"]].append({
        "type": s.get("shape_type"),
        "group_id": s.get("group_id"),
        "description": s.get("description") or "",
        "flags": s.get("flags") or {},
        "points": [[round(x, 6), round(y, 6)] for x, y in pts],   # нормировано к [0, 1]
    })
missing = REQUIRED - shapes.keys()
if missing:
    print("WARNING: missing labels:", sorted(missing))
json.dump({"source_image": src.get("imagePath"), "image_size": [W, H], "shapes": shapes},
          open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print({k: len(v) for k, v in shapes.items()})
```

Сообщения скрипта — на английском: консоль Windows (cp1252) падает на кириллице в `print`.

#### Гигиена данных и лицензий при разметке

- **Разметка только локально или на своём сервере.** Футаж организаторов не загружаем в облачные и бесплатные сервисы:
  - Roboflow Public plan публикует данные;
  - облачные провайдеры в чате X-AnyLabeling не использовать (только локальный Ollama);
  - shared-проекты VIA работают через сервер VGG.
- **Модели для предразметки** (например, YOLO от Ultralytics, AGPL-3.0) — только помощники разметки. В репо они попадают, только если это выбранный детектор, и тогда их надо указать в README.
- **README.** Указать все датасеты и инструменты, которыми делали обучающие данные.

---

## 2. Опенсорс-модели

**Про лицензию репо.** Если берём Ultralytics, **весь публичный репо выходит под AGPL-3.0**: Ultralytics считает модели, обученные его кодом, тоже AGPL. Остальные компоненты должны быть с этим совместимы: Apache-2.0, MIT, BSD, CC0, CC-BY. Не смешивать с весами под non-commercial или no-redistribution: DEIMv2, модели на CrowdHuman, FireViewer, RF-DETR-XL/2XL (PML 1.0), SAM 3 (SAM License).

**Пермиссивная альтернатива без AGPL.** RF-DETR-S/M (Apache-2.0) + пакет `trackers` 2.6.1 (Apache-2.0); в бэкапе D-FINE или RT-DETRv4. Для `road_obstacle` — OmDet-Turbo или MM-GDINO вместо YOLOE. Детектор огня и дыма тогда придётся обучать через `rfdetr`.

### 2.1 Детекторы и трекеры

#### Детекторы

| Модель | Лицензия | Размер весов | Скорость на T4 (таблицы авторов) и точность | Роль в проекте | Рекомендация |
|---|---|---|---|---|---|
| **YOLO26 n/s/m/l/x** (`ultralytics==8.4.163`, выпущен в январе 2026) | AGPL-3.0 (+ Enterprise) | 5.5 / 20.4 / 44.3 / 53.2 / 118.7 MB | TRT10 @640: 1.7 / 2.5 / 4.7 / 6.2 / 11.8 ms; COCO mAP50-95 40.9 / 48.6 / 53.1 / 55.0 / 57.5 (e2e 40.1 / 47.8 / 52.5 / 54.4 / 56.9). @1280 (letterbox 1280x736, ~2.3× пикселей): m ≈ 11 ms TRT FP16 / 20–25 ms PyTorch FP16 (оценка, не подтверждено) | Главный детектор ТС, двухколёсных, людей; fine-tune с COCO на наших кадрах. NMS-free голова (`nms=False`, выход (N, 300, 6)) даёт стабильные боксы для трекинга. COCO-классы traffic light, stop sign, dog, cow, horse, sheep, suitcase пригодятся правилам | **основной** (m; s при дефиците времени) |
| **YOLO11 n–x** | AGPL-3.0 | yolo11m.pt 40.7 MB | TRT10: 1.5 / 2.5 / 4.7 / 6.2 / 11.3 ms; mAP 39.5 / 47.0 / 51.5 / 53.4 / 54.7 | Фолбэк, если у YOLO26 сломается обучение, экспорт или e2e-голова с конкретной версией TensorRT; код тот же | запасной |
| **YOLO26-seg** | AGPL-3.0 | yolo26m-seg.pt 54.8 MB | ~1.4× латентности detect | Маски для точного контакта и пересечения линий | опционально |
| **RF-DETR N/S/M/L** (`rfdetr==1.11.0`) | Apache-2.0 | Сырые чекпойнты N/S/M 366.3 / 386.0 / 405.0 MB (с доп. состоянием), `rf-detr-large-2026.pth` 136.0 MB; очищенный N/S/M ≈130–140 MB (оценка) | AP 48.4 / 53.0 / 54.7 / 56.5 при 2.3 / 3.5 / 4.4 / 6.8 ms (разрешение 384 / 512 / 576 / 704) | Лучший полностью пермиссивный детектор: S на +4.4 AP лучше YOLO26s ценой +1.0 ms; бэкбон DINOv2 хорошо переносится на малые датасеты | запасной (**основной**, если репо не-AGPL) |
| **RF-DETR-Seg N…2XL** | Apache-2.0 (все шесть размеров) | seg-s 135.0 MB, seg-m 143.0 MB | mask AP 40.3 / 43.1 / 45.3 / 47.1 / 48.8 / 49.9 при 3.4 / 4.4 / 5.9 / 8.8 / 13.5 / 21.8 ms (312–768 px) | Пермиссивные маски; низкое входное разрешение (312–504 px у N–L) — используйте кропы | опционально |
| **RF-DETR XL/2XL** (`rfdetr_plus`) | PML 1.0 (требует платформенный план, «Good Standing», телеметрия и учёт использования) | — | — | — | избегать |
| **RT-DETRv4 S/M/L/X** | Apache-2.0 | (не подтверждено); только Google Drive | AP 49.8 / 53.7 / 55.4 / 57.0 при 3.66 / 5.91 / 8.07 / 12.90 ms | Пермиссивный бэкап; учитель DINOv3 нужен только при обучении (гейтед, лицензия DINOv3) | запасной |
| **DEIM** (DEIM-D-FINE / DEIM-RT-DETRv2) | Apache-2.0 (GitHub показывает NOASSERTION из-за нестандартного заголовка LICENSE) | Google Drive | DEIM-D-FINE N–X: AP 43.0 / 49.0 / 52.7 / 54.7 / 56.5 при 2.12 / 3.49 / 5.62 / 8.07 / 12.89 ms | D-FINE, но лучше обучен (+0.2…0.7 AP при той же латентности) | опционально |
| **D-FINE** | Apache-2.0 (не подтверждено в этом проходе) | release 79.1 / 126.1 MB; HF `ustc-community/dfine-small-coco` 41.5 MB | Протокол T4 TRT 10.4 fp16 bs1 (таблица авторов) | Пермиссивный бэкап. Веса `obj2coco` опираются на Objects365 («academic purpose only»), лицензию проверить | опционально |
| **RT-DETR / RT-DETRv2** (PekingU, HF) | (не подтверждено) | 80.9 MB | — | На этом чекпойнте основан accident-детектор dri11heaD (§2.3.2) | опционально |
| **DEIMv2** | Некоммерческая (LICENSE.md) | S 39.4 MB | — | — | избегать |
| **YOLOv10** (THU-MIG) | AGPL-3.0 | n: 11.2 MB safetensors / 9.4 MB onnx | 1.84–10.70 ms (GPU в README не указан) | Идею NMS-free развивает YOLO26 | избегать |
| **YOLOX** | Apache-2.0 | yolox_m.pth 203.1 MB (с train-state) | Только V100 (s 9.8 ms, m 12.3 ms) | Пермиссивный фолбэк; основа светофорного детектора Autoware. pip `yolox 0.3.0` — sdist 2022, ставить из клона | опционально |

#### Трекеры

| Трекер | Лицензия | Веса | Скорость | Роль | Рекомендация |
|---|---|---|---|---|---|
| **Встроенные в Ultralytics** 8.4.163: TrackTrack (по умолчанию), BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, FastTracker | AGPL-3.0 | 0 (ReID `yolo26s-reid.onnx` 28.5 MB скачивается сам только при `with_reid: True`) | CPU < 1 ms/кадр (оценка); GMC `sparseOptFlow` — несколько ms на больших кадрах (оценка) | Основной трекер траекторий: ByteTrack/OC-SORT для скорости, TrackTrack/BoT-SORT для плотных перекрёстков. `track_buffer` 60–90 кадров (по умолчанию 30), чтобы треки переживали перекрытие автобусами | **основной** |
| **supervision 0.30.5** | MIT | 0 | — | `PolygonZone`, `LineZone`, аннотаторы. `sv.ByteTrack` устарел в 0.28.0 и удаляется в 0.31.0 | **основной** (зоны) |
| **Roboflow `trackers` 2.6.1** | Apache-2.0 | 0 | — | Трекер для пермиссивного пути (`ByteTrackTracker`, `OCSORTTracker` и др.); экстра для масок тянет `rf-segment-anything` и `rf-cutie` | запасной (для не-AGPL репо) |
| **BoxMOT 25.0.0** | AGPL-3.0 | ReID-веса скачиваются при первом запуске (размеры не подтверждены) | HOTA на своём сплите MOT17-ablation: OccluBoost 71.10, BoT-SORT 69.68, ByteTrack 67.68, OC-SORT 66.44 (с MOT17-test не сравнимо) | Только если нужен appearance-ReID. Python <3.14; часть ReID обучена на отозванном DukeMTMC | опционально |
| **Эталонные репо:** ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT | MIT | 0 | MOT17 test: BoT-SORT MOTA 80.6 / IDF1 79.5 / HOTA 64.6; ByteTrack 80.3 / 77.3 / 63.1 (не подтверждено); OC-SORT ~700 FPS ассоциации на CPU i9 | Вендоринг кода ассоциации под пермиссивной лицензией | опционально |
| **SORT / StrongSORT** | GPL-3.0 | — | — | StrongSORT: последние новости 2023-06-23 | не нужен |

**Конфиг трекера для неподвижной камеры:**

```bash
pip install lap==0.5.13   # иначе Ultralytics сам ставит lap при первом track() (check_requirements)
```

```yaml
# cfg/bytetrack_fixedcam.yaml — копия bytetrack.yaml; у ByteTrack нет ключей gmc/reid
track_buffer: 90
```

```yaml
# cfg/tracktrack_fixedcam.yaml (или botsort_fixedcam.yaml) — копия исходного yaml, меняем:
gmc_method: none      # компенсация движения камеры не нужна
with_reid: False
track_buffer: 90
# у TrackTrack есть ещё lost_match_thr — для сцен с долгими перекрытиями
```

```python
from ultralytics import YOLO
model = YOLO("weights/best.pt")                 # всегда локальный путь, иначе скачает с GitHub
res = model.track(frame, persist=True, tracker="cfg/bytetrack_fixedcam.yaml", imgsz=1280, half=True)
```

- GMC используют только `botsort.yaml` и `tracktrack.yaml` (по умолчанию `sparseOptFlow`). В `deepocsort.yaml` уже стоит `gmc_method: none`.
- Поведение трекеров меняется от релиза к релизу, поэтому версию пинить.

### 2.2 Светофор, пешеходы, огонь/дым, препятствия, движение, сегментация, декодирование

#### Светофор

| Решение | Лицензия | Размер | Роль | Рекомендация |
|---|---|---|---|---|
| **Фиксированные ROI ламп + HSV/яркость + гистерезис** (OpenCV) | OpenCV — Apache-2.0 (репо упаковки opencv-python — MIT) | 0; ≪1 ms/кадр на CPU (оценка) | Для `red_light`, `stop_line`, контекста `failure_to_yield` | **основной** |
| **Autoware `traffic_light_classifier`** (MobileNetV2, ONNX) | Apache-2.0 | 8.9 MB на файл | Обученный фолбэк для кропов головы светофора (блики, стрелки) | запасной |
| **Autoware fine detector** | Apache-2.0 (не подтверждено для этого файла) | 35.8 MB, статический batch | Детектор светофоров (на базе YOLOX) | опционально |

**Как устроен HSV-вариант.**
- Один раз вручную разметить полигоны ламп в `scene.json`: красная, жёлтая, зелёная, пешеходная.
- На каждом кадре сравнивать V и насыщенность внутри лампы с её базой «выключено», набранной по первым N кадрам.
- Сглаживать гистерезисом или медианой за 0.3–0.5 с.

Риски:
- ШИМ-мерцание LED и rolling shutter — нужно временное сглаживание;
- блики и экспозиция сдвигают пороги;
- светофор нужного направления может смотреть от камеры.

**Autoware classifier.**
- Файлы: `traffic_light_classifier_mobilenetv2_batch_{1,4,6}.onnx` и `ped_traffic_light_classifier_mobilenetv2_batch_{1,4,6}.onnx`. Autoware по умолчанию грузит `batch_6`; batch только фиксированный.
- Вход 224x224, RGB. mean `[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]` (шкала 0–255).
- Классы для машин: green, left,red, left,red,straight, red, red,right, red,straight, unknown, yellow, red,up_left, red,right,straight, red,up_right. Для пешеходов: red, green, unknown.
- Есть поламповый `traffic_light_lamp_recognizer_comlops.onnx`.
- Обучен на кропах с автомобильной камеры TIER IV. Удалённая 4K-камера выглядит иначе — валидировать на dev-сете.

**Про OpenCV.**
- Пинить `opencv-python==4.14.0.94` (2026-07-29). 5.0.0.93 — мажорная версия с миграцией API: удалён legacy C API, calib3d разделён на geometry, calib и stereo.
- В стартовом kit в `requirements.txt` стоит `opencv-python-headless>=4.8`, а `ultralytics` и `trackers` зависят от `opencv-python`. Оба пакета ставят одно пространство имён `cv2`; на PyPI прямо сказано «Do not install multiple different packages in the same environment». В `requirements.txt` оставить **только** `opencv-python==4.14.0.94`.

#### Пешеходы (мелкие на 4K)

- **Тот же детектор + нативные 4K-кропы ROI переходов и тротуаров.** Весов 0. Полный кадр для детектора уменьшаем, а для зон переходов подаём кропы в исходном разрешении, только на тех кадрах, где это нужно.
- **SAHI 0.12.7** — тайловый инференс, поддерживает `yolo26`, `yoloe`, `roboflow` (RF-DETR), `rtdetr`. Лицензия (не подтверждено). Опционально.
- **CrowdHuman** — CC BY-NC 4.0 и условия no-redistribution. Модели на нём в AGPL-репо **избегать**.

#### Огонь и дым (`fire_smoke`)

| Решение | Лицензия | Размер / скорость | Комментарий | Рекомендация |
|---|---|---|---|---|
| **Своя YOLO26s на D-Fire** (+ опционально Pyro-SDIS, FASDD) | Данные: D-Fire CC0-1.0, Pyro-SDIS Apache-2.0, FASDD CC-BY-SA-4.0. Веса AGPL-3.0 (через Ultralytics) | 20.4 MB; 2.5 ms @640 T4 TRT; обучение ~1–2 ч на Kaggle T4, 50–80 эпох (оценка) | Чистая цепочка лицензий. Запускать на каждом 5–10-м кадре (~3–6 Гц), требовать устойчивости и роста ≥2–3 с, можно с гейтом по движению. Hard negatives с dev-сета: выхлоп, пыль, фары, стоп-сигналы, красные машины, блики | **основной** |
| **`rabahdev/fire-smoke-yolov8n`** | AGPL-3.0 | 6.2 MB | Готовый вариант на первый день; обучен на D-Fire (сплит 14 122 / 3 099 / 4 306) | запасной (на день 1) |
| **`pyronear/yolov11s`** | Тег Apache | imgsz 1024 | Лесные пожары; `pyro-vision` в архиве с 2026-08-26 | опционально |
| **Модель SalahALHaismawi (HF)** | Тег MIT; датасет `personal-bodxv/fire-detection-sejra-fognw` — CC BY 4.0, 8 939 изображений | — | Альтернатива для сравнения | опционально |
| **FireViewer** | CHECKPOINT_LICENSE: «You may not redistribute the Model» (включая конвертированные и дообученные версии) | — | Нельзя распространять в публичном репо | избегать |

Команда обучения: `yolo train model=yolo26s.pt data=dfire.yaml imgsz=640 epochs=60` (на Kaggle). D-Fire — в основном уличные и лесные сцены, горящих машин и тёмного выхлопа мало.

#### Препятствия (`road_obstacle`): open-vocabulary и фон

| Решение | Лицензия | Размер | Скорость | Роль | Рекомендация |
|---|---|---|---|---|---|
| **MOG2, двойной фон** (OpenCV) | Apache-2.0 | 0 | CPU | Новый статичный блоб внутри полигона дороги, не являющийся треком ТС, держится ≥N с | **основной** (правило) |
| **YOLOE-26** (l-seg / s-seg) с «запечённым» словарём | AGPL-3.0 | `yoloe-26l-seg.pt` 79.4 MB / `yoloe-26s-seg.pt` 31.1 MB | YOLOE-11-L 130.5 FPS на T4 (README THU-MIG) | Кандидаты «debris / fallen object / animal / box / tire» на ROI с частотой 2–5 Гц | **основной** (детектор) |
| **MM-Grounding-DINO tiny** (и LLMDet tiny), HF transformers | Apache-2.0 | 692.0 MB каждый | ~80–150 ms на изображение 800 px FP16 (не подтверждено) | Верификатор второго этапа на нескольких кропах за видео; офлайн-авторазметка. Zero-shot COCO 50.6 AP; LVIS minival 40.5 / val 30.6 по таблице карточки (в тексте карточки 41.4 / 31.9) | запасной |
| **Grounding DINO (оригинал, Swin-T / Swin-B)** | Apache-2.0 | HF tiny 689.4 MB, base 933.4 MB | ~100–200 ms FP16 (оценка) | Псевдоразметка. Оригинальный репо компилирует CUDA-операции (нужен CUDA_HOME) — на Windows брать HF-версию. T: 48.4 AP zero-shot; у B 56.7 AP не zero-shot | опционально |
| **OWLv2** (`google/owlv2-base-patch16-ensemble`) | Apache-2.0 | 620 MB | вход 960x960 | Альтернативный верификатор | опционально |
| **OmDet-Turbo** | (не подтверждено) | 462 MB | 21.5 / 140 FPS на A100 | COCO 42.5 / LVIS 30.3; в transformers с 4.45 | опционально |
| **Florence-2** (`florence-community`, нативные чекпойнты) | MIT | 463 MB / 1.55 GB | — | Верификатор без `trust_remote_code`; на T4 float16 (быстрого bf16 нет) | опционально |
| **YOLO-World V2.1** | GPL-3.0 («supported for commercial usage») | 94.2 MB | TensorRT «coming soon» | Альтернатива YOLOE | опционально |
| **Grounding DINO 1.5 / 1.6** | Только API | — | — | Облачные API на инференсе запрещены | избегать |
| **SAM 3 / 3.1** | SAM License (всё распространяемое остаётся под SAM License; запреты ITAR, военного использования и др.) — несовместима с AGPL-бандлом; HF с ручным одобрением | 3.44–3.50 GB | медленно на T4 | Только офлайн-авторазметка на Kaggle/Colab; Python ≥3.12, PyTorch ≥2.7, CUDA ≥12.6 | избегать (в пайплайне) |

**Датасеты для препятствий:** RAOD, животные из COCO и Open Images (§3).

**YOLOE офлайн.** Первый вызов `set_classes()` делает pip-установку `git+https://github.com/ultralytics/CLIP.git` и скачивает `mobileclip2_b.ts` (253.8 MB) в текущую папку. Вызывать `set_classes()` **при сборке**, потом сохранить или экспортировать модель; `mobileclip2_b.ts` не отгружать.

#### Движение и оптический поток

| Решение | Лицензия | Размер | Роль | Рекомендация |
|---|---|---|---|---|
| **OpenCV DIS optical flow** @960x540 | Apache-2.0 | 0 | `congestion`: занятость и скорость по полосам + поток | **основной** |
| **RAFT small** (torchvision) | BSD-3-Clause | 4.0 MB (0.99M параметров, 47.66 GFLOPs, EPE 1.9901) | Более точный поток, если DIS не хватит; веса через `TORCH_HOME` | опционально |
| **SEA-RAFT** | BSD-3-Clause | 35.6 MB (S) / 78.8 MB (M) на HF | Лишняя зависимость ради малой пользы | не нужен |

#### Сегментация

- **YOLO26m-seg** (54.8 MB, AGPL-3.0) или **RF-DETR-Seg** (Apache-2.0) — маски для точного контакта и пересечения линий. Опционально.
- **SAM 2.1** — Apache-2.0 (код и веса).
  - Размеры: tiny 38.9M параметров / 156.0 MB; small 46M / 184.4 MB; base+ 80.8M / 323.6 MB.
  - Скорость: 91.2 / 84.8 / 64.1 FPS на A100; на T4 примерно в 4 раза медленнее (не подтверждено).
  - Роль: маски по боксу для разметки и уточнения кадров контакта при авариях. Не для каждого объекта в каждом кадре.
  - `pip sam2 1.1.0` — это sdist, он собирает CUDA-расширение. Сообщение «Failed to build the SAM 2 CUDA extension» можно игнорировать (так сказано в README), на Windows оно ожидаемо.

#### Декодирование видео (вход Part A)

| Решение | Лицензия | Версия | Вывод | Рекомендация |
|---|---|---|---|---|
| **PyAV (`av`)** | BSD-3-Clause. В колёсах FFmpeg DLL с libx264/libx265, то есть GPL-сборка. Как pip-зависимость AGPL-репо это нормально; DLL не вендорить | 18.1.0 (Python ≥3.11); для Python 3.10 нужен `av==17.1.0` (последний с cp310; колёса cp310, cp311, cp314) | Декодирует H.264 4:2:2 10-bit на CPU. Это единственный вариант для T4: NVDEC 4:2:2 не умеет | **основной** |
| **PyNvVideoCodec 2.2.3** | MIT | 2026-09-15 | 4:2:2 декодирует только на Blackwell, поэтому на T4 сэмплы не прочитает. Имеет смысл, только если скрытые тесты окажутся 4:2:0 8-bit | избегать |
| **TorchCodec 0.16.0** | BSD-3-Clause | 2026-08-13 | Нужен системный FFmpeg, а на чистой машине организаторов его может не быть (риск для «Runs as submitted» — 40% оценки за код). CUDA-путь использует NVDEC | избегать |

**Скорость (замер на i5-12450H, синтетика 3840x2160 29.97p H.264 High 4:2:2 10-bit, ~150 Mbps, CAVLC и CABAC с 2 B-кадрами):**

| Способ | Скорость |
|---|---|
| PyAV, полноразмерный bgr24 | ~20 fps |
| PyAV, decode + `frame.reformat` в 1280x720 bgr24 | **~39–50 fps** |
| `cv2.VideoCapture.read()` | ~14–23 fps |
| cv2 `grab()` без конвертации цвета | ~57 fps |

Узкое место — конвертация 10-bit 4:2:2 → BGR в полном разрешении.

```python
import av
c = av.open(path); s = c.streams.video[0]; s.thread_type = "AUTO"
for f in c.decode(s):
    img = f.reformat(width=1280, height=720, format="bgr24").to_ndarray()
    t = float(f.pts * s.time_base)          # точные PTS для границ
```

- Открываем **один раз** и кормим из одного прохода все потребители: детектор, поток, фон.
- Полноразмерные кадры берём только на выбранных кадрах, где нужны нативные кропы пешеходов и светофоров.
- Точный seek работает так: seek на ключевой кадр, потом декод вперёд.
- Цифры получены на синтетике, не на реальных Sony-файлах. CPU машины оценки (8 ядер) может быть медленнее ноутбука (не подтверждено).

#### Рантаймы инференса

- **ONNX Runtime GPU** (MIT).
  - 1.30.0 (2026-09-10): Python ≥3.11, экстры `[cuda]`/`[cudnn]` тянут CUDA 13 (драйвер ≥580).
  - **1.23.2** (2025-10-22): последний с колёсами cp310, сборка CUDA 12 (`nvidia-cuda-runtime-cu12~=12.0`, `nvidia-cudnn-cu12~=9.0`).
  - Роль: портативный раннер маленьких ONNX (классификатор Autoware, экспортированные детекторы) и фолбэк без сборки TRT-движка.
  - `providers=['CUDAExecutionProvider','CPUExecutionProvider']`.
- **TensorRT** (`tensorrt-cu12==11.3.0.99`).
  - Поддерживает SM 7.5 и выше, то есть T4 и GTX 1650 подходят. Сборка cu12 использует CUDA 12.9 («Compatible with CUDA 12.x versions only»). Простой пакет `tensorrt` теперь тянет `tensorrt_cu13`.
  - **Движки непереносимы:** строить на T4 и кешировать. Перенос между GTX 1650 и T4 (не подтверждено). Фолбэк — ONNX Runtime или PyTorch FP16.
  - Экспорт Ultralytics: `YOLO('weights/best.pt').export(format='engine', half=True, imgsz=1280)` на T4. Если tensorrt не установлен, экспорт поставит его сам.
  - Если движок придётся строить на машине оценки, это время, скорее всего, пойдёт в бюджет 3× (не подтверждено). Надёжнее по умолчанию PyTorch FP16 / ONNX Runtime, а TRT включать флагом.

### 2.3 Модели событий (аварии / near-miss), видеоклассификаторы, VLM-верификаторы

**План по приоритету.**

1. **Шаг 0 — без обучаемой модели событий (первые 1–2 дня, все руки).** Детектор + трекер → сглаженные треки → гомография на плоскость дороги (4+ точки, кликнутые вручную; камера одна и неподвижна, делаем один раз на установку камеры).
   - **Кандидаты `accident`:** у двух треков перекрытие боксов или контакт, затем у обоих резкое падение скорости почти до 0, отклонение от направления полосы, и оба стоят. Или трек пропадает / сливается с другим.
   - **Кандидаты `near_miss`:** для пары минимальный 2D-TTC < ~1.0–1.5 с или TAdv/PET < ~1–1.5 с, плюс резкое торможение или манёвр, и без контакта (пороги — эвристика, калибровать на dev-сете).
   - Точность границ (tIoU 0.5/0.7) даёт в основном трек: от первого контакта до момента, когда участники остановились. Это совпадает с конвенцией разметки (§1.2).
2. **Шаг 1 — обучаемый классификатор клипов** (~6–10 человеко-часов с подготовкой данных, 2–4 ч GPU на Kaggle; оценка). VideoMAE V2 distilled ViT-S/16 → классы accident / near_miss / normal (опционально fire_smoke), fp16 AMP.
   - **Вход:** 16 кадров при 5–8 fps: (a) полный кадр в 224, (b) ROI-кроп вокруг взаимодействующей пары, вырезанный из 4K-исходника (это главное преимущество неподвижной 4K-камеры).
   - **Данные:** ACCIDENT (реальные CCTV + CARLA), TAD, AIC21-T4, CADP, TADBench (§3), плюс hard negatives с наших видео: плотные очереди, перекрытия, остановки автобусов.
   - **Инференс:** шаг 0.25–0.5 с → скор по времени → слияние со скором правил (логрегрессия или GBDT на признаках dev-сета) → пороги с гистерезисом → границы «прилипают» к событиям треков.
3. **Шаг 2 — если останется время:** (a) ViT-B вместо ViT-S; (b) V-JEPA 2.1 ViT-B (заморожен) + attentive probe как второе мнение, ансамбль скоров; (c) модель по траекториям (маленький GRU/Transformer или GBDT на TTC/DRAC/TAdv/изменении скорости) для near_miss против normal — обучается только на наших треках, лицензий весов нет; (d) zero-shot подсказки Cosmos-Embed1-anomaly или SigLIP2 для кандидатов `road_obstacle` / `fire_smoke`.

**Part B (причинный риск «авария начнётся в ближайшие 5 с»).** Только прошлые кадры: `RiskEstimator.step` может использовать только уже полученные кадры.
- Risk(t) = откалиброванный максимум по парам f(min TTC, DRAC, скорость сближения, TAdv) + каузальное окно VideoMAEv2-S, заканчивающееся в t, обученное на метках «авария начинается в течение 5 с» (рецепт как у TimeSouth, arXiv 2606.09542; реализовать самим — у их весов нет лицензии).
- Сглаживание только каузальным EMA. Детерминизм обязателен (фиксировать seed).
- Тяжёлую работу делать раз в N кадров, на остальных — дешёвое обновление состояния (см. бюджет времени §2.3.6).
- MoViNet-Stream — опциональная крошечная каузальная альтернатива.

**Свидетельства из литературы (arXiv проверены):**
- SynCrash (arXiv 2608.29759, 2026-08-30): VideoMAEv2-giant, дообученный на синтетике CARLA, для временной локализации в ACCIDENT@CVPR 2026 (неподвижные CCTV).
- TimeSouth (arXiv 2606.09542): VideoMAE-v2 Base — 2-е место в CVPR 2026 AUTOPILOT (zero-shot accident anticipation, dashcam).
- arXiv 2608.08867: training-free двухпроходный пайплайн Qwen3-VL-32B + YOLO11x + BoT-SORT с оверлеями ID треков обошёл все бейзлайны организаторов на реальных CCTV в ACCIDENT@CVPR 2026 (harmonic mean 0.504 против 0.412).
- arXiv 2604.09685: zero-shot пайплайн ACCIDENT@CVPR 2026: CLIP-сходство для типа столкновения + пики разности кадров и поток Farneback для времени и места.

#### 2.3.1 Видео-бэкбоны и головы временной локализации

| Модель | Лицензия | Размер | Скорость на T4 | Роль | Рекомендация |
|---|---|---|---|---|---|
| **VideoMAE V2 distilled ViT-S/16 (K710)**: `OpenGVLab/VideoMAE2` → `distill/vit_s_k710_dl_from_giant.pth` | Код MIT; HF-репо с тегом `apache-2.0` (README — заглушка на 60 байт). **Не путать** с `OpenGVLab/VideoMAEv2-Base` (cc-by-nc-4.0). Данные K710 (Kinetics, YouTube) указать в README | ~22M параметров, 44.3 MB (44 334 609 байт) | ~57 GFLOPs на клип 16×224; 150–300 клипов/с fp16 (оценка по FLOPs, не подтверждено) | Главная обучаемая модель accident / near_miss; каузальная версия — для Part B. MODEL_ZOO: K710 top-1 77.6, K400 83.7, K600 83.1 (K400/K600 — после доп. дообучения) | **основной** |
| **VideoMAE V2 distilled ViT-B/16 (K710)** (`distill/vit_b_k710_dl_from_giant.pth`) | Как у ViT-S | ~87M, 173.6 MB | ~180 GFLOPs; 60–110 клипов/с (оценка) | Апгрейд после работающего S: K710 81.5 / K400 86.6 / K600 85.9 (+2.9 top-1 на K400). Легче переобучается, обучение ~3× медленнее | запасной |
| **V-JEPA 2 / 2.1** (ViT-B 80M, ViT-L 300M) + attentive probe | MIT (большая часть кода; несколько файлов — отдельные условия); HF `facebook/vjepa2-*` — MIT; зеркало `apiantonio/vjepa2.1-vit-base-384` — MIT | 2.1 ViT-B/16: 80M по таблице README; зеркало хранит 109.7M F32 (0.44 GB), вероятно не только энкодер (не подтверждено). `vjepa2-vitl-fpc16-256-ssv2`: 375.5M, 1.50 GB fp32 | 16 кадров × 384 px = 4 608 токенов → ~10–20 клипов/с (оценка); на 256 px быстрее | Второе мнение / ансамбль: замороженный энкодер + лёгкий probe, признаки кешировать. Веса 2.1 только на `dl.fbaipublicfiles.com` (через `torch.hub.load('facebookresearch/vjepa2','vjepa2_1_vit_base_384')`) и в зеркалах; для `download.sh` зеркалировать файл. V-JEPA 2 — в transformers 5.17.0 | запасной |
| **X3D XS/S/M/L** (PyTorchVideo) | Код Apache-2.0; веса обучены на Kinetics-400, отдельная лицензия весов не указана | 3.79M (XS/S/M), 6.15M (L); десятки MB | 0.91 / 2.96 / 6.72 / 26.64 GFLOPs на view; K400 top-1 69.12 / 73.33 / 75.94 / 77.44 | Дешёвый постоянный скорер движения или каузальная голова Part B на кропах. **PyPI 0.1.5 сломан** с новым torchvision (`torchvision.transforms.functional_tensor`) → `torch.hub.load('facebookresearch/pytorchvideo', 'x3d_s', pretrained=True)` или `pip install git+https://github.com/facebookresearch/pytorchvideo` | опционально |
| **MoViNet A0–A2 Stream** | MoViNet-pytorch — MIT; `kfkas/movinet-a0-stream-pytorch` — MIT; `litert-community/MoViNet-A0-Stream-LiteRT` — apache-2.0; исходные TF-веса — TF Model Garden (Apache-2.0) | A0: 3.77M, ~15 MB fp32 | K600 top-1: A0 72.05, A1 76.45, A2 78.40 (172/172/224 px) | Part B: каузальная по построению, покадровый стриминг (`reset_stream()`, `use_stream_state=True`). Порты неофициальные (Atze00 заброшен с 2022; kfkas — новый, один автор) | опционально |
| **UniFormerV2** | Код Apache-2.0; HF-веса `Andy1621/uniformerv2` — MIT; инициализация OpenAI CLIP (MIT) | В HF два чекпойнта B/16 8x224 по 0.46 GB fp32 (0.92 GB) | — | Альтернатива; завязан на экосистему SlowFast / mmaction | опционально |
| **InternVideo2 distilled S14 / B14 / L14** | GitHub-репо Apache-2.0; HF-репо дистилляции **без тега** лицензии (отметить в README); `InternVideo2_CLIP_S` — apache-2.0 | S14 ~47 MB, B14 ~180 MB, L14 ~619 MB | — | Альтернативный бэкбон. Zero-shot по тексту — только через отдельные репо `InternVideo2_CLIP_*`. Интеграция дольше, чем у VideoMAEv2 | опционально |
| **ActionFormer** / **TriDet** / **OpenTAD** (головы временной локализации) | MIT / MIT / Apache-2.0; предобученные веса не нужны | Голова — несколько M параметров (не подтверждено) | Ничтожно; обучается на кешированных признаках за минуты | Прямая регрессия `[start, end, class]` по признакам сниппетов (VideoMAEv2-S при 4–8 сниппетах/с + TTC/скорость) для accident / near_miss / fire_smoke / congestion. TriDet: THUMOS14 mAP 83.6/80.1/72.9/62.4/47.4 при tIoU 0.3–0.7; ActionFormer: 71.0% mAP@0.5. Нужно много размеченных событий, иначе переобучение. C++ NMS ActionFormer (Linux, PyTorch 1.11, numpy<=1.23) на Windows/3.11+ заменить чистым PyTorch 1D NMS | опционально |
| **VideoMAE v1** (MCG-NJU, `videomae-*-finetuned-kinetics`) | CC-BY-NC-4.0 (тег HF и файл LICENSE) | base ~0.35 GB fp32 | — | Удобен только классом `VideoMAEForVideoClassification`; V2 distilled сильнее и чище по лицензии | избегать |
| **TimeSformer** (`facebook/timesformer-*`) | CC-BY-NC-4.0; GitHub в архиве | base ~0.49 GB | — | Старше и слабее VideoMAEv2 / V-JEPA | избегать |

Также проверены (без изменений), но для нас не в приоритете: SlowFast (R50: 34.57M, 65.71 GFLOPs, 76.94; R101 16x8: 53.77M, 78.70), видеомодели torchvision, Video Swin (Apache-2.0, последний push 2023-03-08), MMAction2 (1.2.0 от 2023-10-12).

**Как дообучать VideoMAEv2-S.**
- Вендорить `models/modeling_finetune.py` (MIT, с атрибуцией) и `pip install timm`. Файл импортирует `timm.models.layers` и `timm.models.registry`: они устарели, но ещё есть в timm 1.0.30 (выдают FutureWarning).
- `vit_small_patch16_224(num_classes=3 или 4, all_frames=16, tubelet_size=2)`, загрузить `vit_s_k710_dl_from_giant.pth` без головы, fine-tune в AMP fp16 с layer-wise LR decay.
- Внимание — явный softmax, без FlashAttention / xformers, поэтому fp16 на Turing работает. Замена на `F.scaled_dot_product_attention` — небольшая правка, экономит память (опционально).
- Сохранить fp16 safetensors (~44 MB).

#### 2.3.2 Готовые «аварийные» чекпойнты и anomaly-модели — почему не отгружаем

| Модель | Лицензия | Размер | Вывод | Рекомендация |
|---|---|---|---|---|
| **TimeSouth zero-shot-taa** (2-е место CVPR 2026 AUTOPILOT; VideoMAE-v2-B, per-frame риск) | **Лицензии нет** ни в GitHub, ни на HF → все права защищены. Данные — Nexar collision prediction под Nexar Open Data License (атрибуция; запрет перепродажи, реидентификации, разработки оружия и вредных систем) | `epoch_14.pth` 1.045 GB, 87M | Референс дизайна Part B: скользящие окна, per-frame голова, клипы 5 с × 150 кадров из Nexar. Домен — dashcam. Веса не отгружать | опционально (как идея) |
| **Cosmos-Embed1-448p-anomaly-detection** (NVIDIA) | Тег HF `other`: NVIDIA Open Model License (коммерция и распространение производных с атрибуцией), доп. информация Apache-2.0 и MIT | 1.198B, 4.79 GB F32 (~2.4 GB при хранении в fp16) | Zero-shot эмбеддинги видео-текст: 8 кадров при 1–2 fps, 448 px → 768-d, косинус к промптам. 24 категории, среди них Traffic Accidents, Obstacles on Road, Animals Obstructing Traffic, Falling Objects, Fire, Pedestrian Jaywalking, Red Light Violation, Wrong-Way Driving, Illegal Lane Changing, Illegal Parking. На своём тесте Vad-Reasoning: macro-F1 38.94%, top-1 hit rate 46.44%. Карточка: старые архитектуры могут требовать FP32 → на T4 fp32 (~4.8 GB VRAM). Источники обучения (UCF-Crime, XD-Violence, TAD, ShanghaiTech, UBnormal, ECVA, интернет-видео) — в README. `trust_remote_code=True`. Конкурирует с VLM за бюджет 5 GB | опционально |
| **DETR / RT-DETR accident-детекторы** (`hilmantm/detr-traffic-accident-detection`, `dri11heaD/rtdetr-vehicle-accident-detection`) | Apache-2.0 (теги HF); лицензия обучающего Roboflow-датасета (не подтверждено) | 41.6M / 42.9M, ~0.17 GB каждый | Одиночный кадр → плохие границы. У DETR нет метрик; у RT-DETR авто-карточка: mAP 0.44, mAP50 0.60, accident mAP 0.587 на своём вал-сплите | избегать |
| **Enos-123 YOLO11x** | Тег MIT, но база — Ultralytics YOLO11 (AGPL, неоднозначно) | — | mAP50 0.826, accident F1 0.833 (карточка); одиночный кадр | избегать |
| **Community VideoMAE accident / UCF-Crime fine-tunes** (esmaelehab, OPear, jatinmehra, akhra92) | esmaelehab CCTV-3s-2 — без лицензии; esmaelehab Accident-dataset и OPear — CC-BY-NC-4.0; jatinmehra — тег MIT на базе VideoMAE-L под CC-BY-NC (сомнительно); akhra92 (4 репо) — MIT | 0.34 GB (esmaelehab), 1.22 GB (jatinmehra, OPear; репо OPear 3.65 GB с `optimizer.pt`) | Нет данных/метрик, dashcam или UCF-Crime, NC-наследование. Максимум — сравнение warm-start на dev-сете | избегать |
| **VadCLIP / RTFM / UR-DMU** (weakly-supervised VAD) | Apache-2.0 / лицензии нет / MIT | Малые головы над CLIP/I3D-признаками; веса на Baidu / OneDrive | Общие аномалии на криминальных видео; плохие границы при tIoU 0.5/0.7 | избегать |
| **HolmesVAU-2B, LaGoVAD/PreVAD, UString, DSTA, GSC, CRASH** | MIT (HolmesVAU, UString, DSTA); Apache-2.0 (LaGoVAD); GSC — без лицензии; код CRASH не найден | HolmesVAU 2.2B BF16, «<9GB memory» | Старые PyTorch 1.x, ego-view, или не влезают в бюджет. Только идеи | избегать |

#### 2.3.3 Суррогатные меры безопасности (TTC / PET) — для near_miss и Part B

| Инструмент | Лицензия | Что даёт | Рекомендация |
|---|---|---|---|
| **Two-Dimensional-Time-To-Collision** (Yiru-Jiao) | MIT | 2D TTC (ТС как ориентированные прямоугольники), DRAC, MTTC; векторизовано: ~0.036 с на 1e4 пар, 0.43 с на 1e5, 7.17 с на 1e6. Не на PyPI: копировать `src/TwoDimTTC.py` и `src/TwoDimSSM.py`. Вход — DataFrame с `x_i, y_i, vx_i, vy_i, hx_i, hy_i, length_i, width_i` (и то же для j) | **основной** |
| **SSMsOnPlane** (тот же автор) | MIT | TTC / DRAC / MTTC / PSD / TAdv / ACT / TTC2D; push 2026-07-03 | **основной** (вместе с предыдущим) |
| **Emergency Index (EI / MEI)** | MIT | Глубина вторжения в зону безопасности — доп. признак тяжести near_miss. MEI — отдельный репо `AutoChengh/MEI` (содержимое не подтверждено) | опционально |
| **Traffic Intelligence** (N. Saunier, Bitbucket) | MIT | Референс: вероятность столкновения по прогнозу движения, TTC, PET. C++-трекер на Windows собирать не нужно; при необходимости вендорить отдельные Python-функции с атрибуцией | опционально |
| **LibSSM** | MIT | TTC, Gap Time, PET и др., но «still under development», неактивен с 2023 | избегать |

Требования и риски: нужны метрические координаты на плоскости дороги, курсы и размеры → ошибки гомографии и дрожание треков критичны (сначала сглаживать треки). TTC с постоянной скоростью плохо описывает медленные осознанные взаимодействия (README). При лёгком перекрытии боксов появляются крошечные положительные значения.

#### 2.3.4 VLM-верификаторы

**По умолчанию — no-go на критическом пути.** Условный go — только как re-scorer кандидатов `accident` под флагом.

| Модель | Лицензия | Размеры (проверено) | В 5 GB | Замечания | Рекомендация |
|---|---|---|---|---|---|
| **Qwen3-VL-2B-Instruct** | Apache-2.0 (официальный GGUF — тоже) | 2.13B; bf16 4.26 GB. GGUF: Q4_K_M 1.107, Q8_0 1.834, F16 3.447 GB; mmproj F16 0.819 / Q8_0 0.445 GB. AWQ-4bit (cyankiwi) 2.23 GB. FP8 3.47 GB (на T4 бесполезен) | GGUF Q8_0 + mmproj Q8 = 2.28 GB; Q4_K_M + mmproj Q8 = 1.55 GB; bf16 — слишком впритык | **Слабо судит столкновения:** AV Collision 36.33% против 74.33% у Cosmos-Reason2-2B (карточка Cosmos). В transformers с 4.57.0 (заметка карточки про установку из исходников устарела). bf16-обучена → переполнение fp16 на T4 (не подтверждено; проверить рано). Или llama.cpp: `llama-mtmd-cli` / `llama-server -m Qwen3VL-2B-Instruct-Q8_0.gguf --mmproj mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` | запасной |
| **Cosmos-Reason2-2B** (NVIDIA; Qwen3-VL-2B, дообученный для physical AI) | NVIDIA Open Model License (коммерция, производные можно создавать и распространять) + доп. условия Apache-2.0; HF gated с автоодобрением | 2.44B, 4.88 GB bf16. `embedl/Cosmos-Reason2-2B-W4A16` 2.79 GB (отдельная лицензия embedl, gated). Community GGUF (не подтверждено) | Только своя квантованная копия (GGUF Q8_0 — разрешённая производная) с атрибуцией | Самый доменный: AV Collision 74.33%. Тестирован только в BF16 на H100/A100, минимум 24 GB → fp16 на T4 (не подтверждено). Карточка советует видео fps=4. Gated-загрузка усложняет `download.sh` | запасной |
| **MiniCPM-V 4.6** (1.3B: SigLIP2-400M + Qwen3.5-0.8B) | Apache-2.0 | bf16 2.60 GB; официальный AWQ 1.87 GB; GGUF Q4_K_M 0.53 + mmproj f16 1.11 = 1.64 GB | Да | Нативный класс transformers ≥5.7.0 (`AutoModelForImageTextToText`, без `trust_remote_code`). Видео до 128 кадров. LLM 0.8B — может быть слаб в оценке столкновений. Gated DeltaNet: в transformers 5.17 есть PyTorch-фолбэк, скорость на sm75 (не подтверждено). Карточка: torchcodec может падать с CUDA 13.1 из torch>=2.11 → PyAV | запасной |
| **Qwen3.5-0.8B / 2B / 4B** | Apache-2.0 | 0.8B: 1.75 GB bf16. 2B: 4.55 GB bf16; GGUF Q4_K_M 1.28 + mmproj F16 0.67 = 1.95 GB; AWQ 2.50 GB. 4B: GGUF 2.74 + 0.67 = 3.41 GB; AWQ 4.04 GB | 2B GGUF — да | Гибрид (Gated DeltaNet + gated attention); PyTorch-фолбэк есть, скорость на T4 (не подтверждено). Thinking выключать (`enable_thinking=False`): 2B склонна к петлям. Рискованнее Qwen3-VL на Turing | запасной |
| **Qwen3-VL-4B** | (лицензия в этом прогоне не выписана) | GGUF Q4_K_M + mmproj Q8 = 2.95 GB; AWQ (cyankiwi) 4.43 GB | Только GGUF через llama.cpp | Если останется запас бюджета | опционально |
| **InternVL3.5-1B** | Apache-2.0 | bf16 2.12 GB | Да | transformers ≥4.52.1 | опционально |
| **MiniCPM-V 4 / 4.5** | Apache-2.0 | V4: 4.06B, int4 2.77 GB (bitsandbytes NF4); V4.5: int4 6.53 GB | V4 — да; V4.5 — нет | Нужен bitsandbytes на инференсе; вытеснен V4.6 | избегать |
| **Phi-4-multimodal / Phi-3.5-vision** | MIT | 12.8 GB (репо) / 8.29 GB bf16 | Нет | Нет нативного видео | избегать |
| **Gemma 4, Qwen2.5-VL, HolmesVAU, всё ≥7B** | Разные (Qwen2.5-VL-3B — qwen-research) | — | Нет | Не влезают или лицензия | избегать |

**Zero-shot энкодеры (дешёвые подсказки на кропах: «smoke», «fire», «debris on road», «animal on road», «crashed car»)** — опционально:
- SigLIP2 `google/siglip2-base-patch16-224` — Apache-2.0; 375M (большая мультиязычная текстовая башня), 1.5 GB fp32 / ~0.75 GB fp16.
- PE-Core `facebook/PE-Core-S16-384` — Apache-2.0; 0.35 GB; нужен код Meta `perception_models`, а не чистый transformers (не подтверждено).
- X-CLIP `microsoft/xclip-base-patch32` — MIT; 196.6M, 0.79 GB fp32 (репо 1.57 GB из-за .bin + .safetensors).
- Для временных событий слабы; пороги и промпты подбирать на dev-сете.

**Рантаймы VLM на T4.**
- **transformers 5.17.0 + SDPA fp16** (Apache-2.0) — наименее рискованный путь для Qwen3-VL-2B / InternVL3.5-1B / MiniCPM-V-4.6. FlashAttention-2 не нужен.
- **llama.cpp** (MIT) — нужен для GGUF (Qwen3-VL-4B влезает только так). Для офлайн-машины — готовые release-бинарники с CUDA. Поддержка Qwen3-VL mtmd в `llama-cpp-python` 0.3.35 (не подтверждено), release-бинарники надёжнее.
- **vLLM 0.30.0** (Apache-2.0) — T4 (CC ≥7.5) поддерживается, но только Linux (на ноутбуке — через WSL), плюс время старта и преаллокация памяти рядом с детектором.
- **У T4 нет tensor cores для bf16 и FP8:** FP8-чекпойнты бесполезны, bf16-чекпойнты гонять в fp16 (или fp32 для малых частей). Ядра compressed-tensors W4A16 (AWQ cyankiwi) на sm75 (не подтверждено).
- Стек тестировать на Kaggle T4 (тот же класс GPU); версии и индекс CUDA-колёс пинить в `requirements.txt`.

**Критерии go** (решаем в фиксированной точке, например когда Шаг 1 оценён на dev-сете):
1. Ограничивает именно precision `accident` на dev-сете: много ложных срабатываний при нормальном recall.
2. На ≥40 размеченных кандидатах (сбалансированно) VLM даёт ≥0.8 balanced accuracy по логиту P("Yes"), порог подобран кросс-валидацией.
3. ≤3 с на запрос на Kaggle T4 в fp16, без NaN и мусора.
4. Суммарные веса ≤5 GB.

Если хоть один критерий не выполнен — no-go. Верификатор не лечит ошибки границ (а они доминируют при tIoU 0.5/0.7), он только убирает ложные срабатывания.

**Если go:**
- **Порядок моделей:** A/B-тест Qwen3-VL-2B-Instruct и Cosmos-Reason2-2B на кандидатах dev-сета; затем MiniCPM-V-4.6 (1.6–2.6 GB); Qwen3-VL-4B GGUF Q4 (2.95 GB), если есть запас.
- **Протокол промпта:** 6–12 кадров из [t−2 с, t+3 с], ROI-кроп из 4K, на кадрах нарисованы ID и боксы треков. Закрытый вопрос («Do vehicle 3 and vehicle 7 physically collide? Answer Yes or No.»), `max_new_tokens=1`, читаем вероятности токенов Yes/No. Лимит ~10–20 кандидатов на видео; при 1–3 с на вызов (оценка) это меньше минуты на видео.

#### 2.3.5 Лимит весов 5 GB

| Компонент | Выбор | Размер |
|---|---|---|
| Детектор | дообученный YOLO26m `best.pt` (s — если нужен запас по времени) | ~0.044 GB (s: ~0.020) |
| Трекер, зоны, HSV-светофор, MOG2, DIS | — | 0 |
| `fire_smoke` | YOLO26s на D-Fire | ~0.020 GB |
| `road_obstacle` | YOLOE-26l-seg с запечённым словарём (или s) | 0.079 GB (0.031) |
| Светофор (фолбэк) | Autoware classifier ONNX | 0.009 GB |
| accident / near_miss | VideoMAEv2-S fp16 (ViT-B: 0.174 GB) | 0.044 GB |
| Опц. второе мнение | V-JEPA 2.1 ViT-B + probe (fp16) | ~0.22 GB |
| Опц. верификатор кропов | MM-GDINO-T (или OWLv2 0.62 GB) | 0.692 GB |
| Опц. VLM | Qwen3-VL-2B GGUF Q8_0 + mmproj Q8 / MiniCPM-V-4.6 GGUF | 2.28 / 1.64 GB |

- **Ядро** (.pt): ~0.15–0.21 GB; с ONNX-копиями ~0.45 GB.
- **+ open-vocabulary верификатор** — ~1.1 GB.
- **+ VLM** — ~3.4 GB, остаётся >1.5 GB запаса.
- **Не влезают разумно:** SAM 3 (3.4 GB сам по себе), Qwen3-VL-2B в bf16 (4.26 GB) вместе с остальным, всё ≥7B. Cosmos-Embed1 в fp16 (~2.4 GB) конкурирует с VLM — выбирать одно.

#### 2.3.6 Бюджет времени: 3× длительности на Part A + Part B вместе

**Критическая находка.** `run_submission.py` (Part B) **после** возврата `detect_events` прогоняет **каждый** кадр через `cv2.VideoCapture.read()` в полном разрешении, шаг 1, в том же бюджете 3×.
- На ноутбуке (i5-12450H) синтетика 4K 29.97p H.264 High 4:2:2 10-bit ~150 Mbps читается cv2 4.13 всего на **~14–23 fps** (0.45–0.75× real-time). Бенчмарки других агентов в стиле харнесса (`sweep_all.csv`) согласуются: ~18–20 fps, 0.65× real-time.
- Сам декод (`grab`) ~57 fps; узкое место — конвертация 10-bit 4:2:2 → BGR в полном разрешении.
- **Итог:** если скрытые тесты в формате сэмплов, собственный Part B харнесса съедает **~1.3–2.2× длительности**, на весь Part A остаётся **~0.8–1.7×**, а на 8-ядерном CPU сервера может быть меньше (не подтверждено).
- Закреплённый `opencv-python==4.14.0.94` не бенчмаркался (установка не успела).

**Пример для 6-минутного клипа** (оценки):

| Статья | Время |
|---|---|
| Лимит 3× | 18 мин |
| Part B харнесса (декод cv2 полного 4K) | ~7.8–13.2 мин |
| Остаток на Part A | ~4.8–10.2 мин |
| Цель для Part A (≤0.6×) | ≤3.6 мин |
| Декод Part A через PyAV + `reformat` в 720p (39–50 fps) | ~3.6–4.6 мин (0.6–0.77×) — **должен идти параллельно с GPU** |
| YOLO26m @1280 на каждом 2-м кадре (~5.4k кадров): ~11 ms TRT FP16 / 20–25 ms PyTorch FP16 | ~1–2.5 мин |
| fire/smoke 3–6 Гц, YOLOE 2–5 Гц на ROI | < +10% |
| VideoMAEv2-S, ~720 окон | секунды |
| VLM (если go), 10–20 вызовов × 1–3 с | < 1 мин |

**Что делать:**
1. Part A декодирует **один раз** через PyAV + swscale-уменьшение (замерено ~39–50 fps), декод в отдельном потоке/процессе параллельно с GPU; цель Part A ≤ ~0.6× длительности.
2. `RiskEstimator.step` почти бесплатный на большинстве кадров (тяжёлое — раз в N кадров, иначе дешёвое обновление состояния).
3. Спросить организаторов формат скрытых тестов: PDF пишет «typical clips are several minutes long at 25 fps», сэмплы — 29.97p 4K 4:2:2. fps всегда читать из файла, не хардкодить.
4. **Рано** замерить полное wall-time `run_submission.py` на сэмплах.
5. Детерминизм (фиксированные seed), `RiskEstimator.step` — только по уже полученным кадрам.

### 2.4 Итоговый bill of materials

**Python.** Для разработки и упаковки — 3.11 (или 3.12, проверено на этом ноутбуке через `uv venv --python 3.12`); оценка обещает «3.10+», поэтому один раз прогнать на 3.10. Колёса `torch 2.14.0+cu126` есть только для cp310–cp313, так что системный Python 3.14 ноутбука для пайплайна не подходит. Совместная резолюция всего набора ниже не проверялась — прогоните `uv pip compile` для 3.10 и 3.11.

#### pip-пакеты (версии проверены на PyPI 2026-09-25)

```text
--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.14.0+cu126          # дефолтный torch 2.14 с PyPI под Linux — CUDA 13 (драйвер >=580); cu126 работает с драйвером >=525
torchvision==0.29.0+cu126
ultralytics==8.4.163         # AGPL-3.0; патч-релизы почти ежедневно — пинить
lap==0.5.13                  # иначе Ultralytics ставит lap сам при первом track()
opencv-python==4.14.0.94     # ОДИН пакет cv2; без opencv-python-headless (он в requirements стартового kit — убрать)
supervision==0.30.5          # MIT; sv.ByteTrack устарел, удаляется в 0.31
numpy==2.2.6                 # последний с колёсами cp310 (есть и cp311, cp312)
scipy==1.15.3                # последний с колёсами cp310
av==17.1.0                   # PyAV; последний с cp310 (на 3.11+ можно 18.1.0)
onnx==1.23.0
onnxslim==0.1.96             # только для экспорта
timm==1.0.30                 # для modeling_finetune.py VideoMAEv2 (FutureWarning про timm.models.layers)
# опционально
onnxruntime-gpu==1.23.2      # последний с cp310, сборка CUDA 12; 1.30.0 требует Python >=3.11 и тянет CUDA 13
tensorrt-cu12==11.3.0.99     # только если строим движки на T4; простой 'tensorrt' теперь тянет сборку cu13
sahi==0.12.7
trackers==2.6.1              # Apache-2.0, для пермиссивного пути
rfdetr==1.11.0               # Apache-2.0; тянет transformers>=5.1,<6
transformers==5.17.0         # только для MM-GDINO / OWLv2 / Florence-2 / VLM
```

- `ultralytics` 8.4.163 на Python ≥3.11 ставит ещё `ultralytics-platform` 0.1.65 (AGPL-3.0, ленивый импорт, зависит от `httpx`).
- `rfdetr[onnx]` пинит `onnxruntime<1.24` на Python 3.10.
- BoxMOT 25.0.0 на Python 3.14 не ставится (требует <3.14).
- `sam2 1.1.0`, `yolox 0.3.0` и `tensorrt-cu12` — sdist (`tensorrt-cu12` — мета-пакет, подтягивающий платформенные колёса).
- Для Two-Dimensional-TTC нужен `pandas` (версию закрепить; не проверялась).

#### Файлы весов (URL и размеры проверены HTTP HEAD 2026-09-25)

| Файл | Размер | Назначение | Лицензия |
|---|---|---|---|
| `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt` | 44.3 MB | Старт для дообучения (отгружаем свой `best.pt`); зеркало HF `Ultralytics/YOLO26` | AGPL-3.0 |
| `.../v8.4.0/yolo26s.pt` | 20.4 MB | Старт для fire/smoke или облегчённого детектора | AGPL-3.0 |
| `.../v8.4.0/yolo26m-seg.pt` | 54.8 MB | Опц. маски | AGPL-3.0 |
| `.../v8.4.0/yoloe-26l-seg.pt` / `yoloe-26s-seg.pt` | 79.4 / 31.1 MB | `road_obstacle` (словарь запечь) | AGPL-3.0 |
| `.../v8.4.0/mobileclip2_b.ts` | 253.8 MB | Только при сборке для `set_classes` YOLOE; **не отгружать** | — |
| `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt` | 40.7 MB | Фолбэк-детектор | AGPL-3.0 |
| `https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf` | 0.77 MB | Только если рисуем результаты | — |
| `https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt` | 6.2 MB | fire/smoke на день 1 | AGPL-3.0 |
| `https://huggingface.co/AutowareFoundation/traffic_light_classifier/resolve/main/traffic_light_classifier_mobilenetv2_batch_1.onnx` | 8.9 MB | Фолбэк классификатора светофора | Apache-2.0 |
| `https://huggingface.co/OpenGVLab/VideoMAE2` → `distill/vit_s_k710_dl_from_giant.pth` | 44.3 MB | Старт для модели аварий (отгружаем свой fp16) | MIT / apache-2.0 |
| `https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth` | 4.0 MB | Опц. поток | BSD-3-Clause |
| HF `openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det` | 692 MB | Опц. верификатор | Apache-2.0 |
| HF `google/owlv2-base-patch16-ensemble` | 620 MB | Опц. верификатор | Apache-2.0 |
| `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt` | 156.0 MB | Только разметка | Apache-2.0 |
| `https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth` | 386.0 MB сырой (~130 MB после очистки, оценка) | Пермиссивная альтернатива | Apache-2.0 |
| `https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco.pth` | 79 MB | Пермиссивная альтернатива (Objects365 в предобучении — проверить) | (не подтверждено) |

**Суммарно:** ядро ~0.15–0.21 GB в .pt, ~0.45 GB с ONNX-копиями; с одним open-vocabulary верификатором ~1.1 GB; остаётся >3.5 GB под VLM или модель аварий. Все веса — только по локальным путям, `weights/download.sh` качает их заранее.

#### Офлайн-подводные камни (проверено по исходникам ultralytics 8.4.163 и rfdetr 1.11.0)

```python
# в самом начале точки входа, ДО импорта ultralytics / transformers / torch.hub
import os
REPO = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_OFFLINE", "1")          # is_online() -> False
os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # без pip-установок lap / tensorrt / CLIP на лету
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(REPO, ".ultralytics"))  # должен быть доступен на запись; Arial.ttf — сюда
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.join(REPO, "weights", "hf"))
os.environ.setdefault("TORCH_HOME", os.path.join(REPO, "weights", "torch"))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO, ".mplcache"))  # первый импорт matplotlib строит кеш шрифтов
# после импорта: from ultralytics import settings; settings.update({"sync": False})  # анонимная телеметрия off
```

- **Веса Ultralytics.** `YOLO('yolo26m.pt')` при отсутствии файла качает с GitHub release v8.4.0 — всегда явные локальные пути.
- **ReID трекера.** С `with_reid: True` и моделью `yolo26*-reid.onnx` файл скачивается сам. Держать `with_reid: False` или отгрузить файл.
- **YOLOE.** Первый `set_classes()` ставит `git+https://github.com/ultralytics/CLIP.git` и качает `mobileclip2_b.ts` в текущую папку. Вызвать при сборке, сохранить/экспортировать модель.
- **TensorRT-экспорт** сам ставит `tensorrt`, если его нет. Движки непереносимы — строить на T4 и кешировать; фолбэк ONNX Runtime / PyTorch.
- **RF-DETR.** Создание модели качает с `storage.googleapis.com`, если не передать `pretrain_weights=<локальный .pth>`; с ним пропускается и загрузка DINOv2 из hub.
- **Hugging Face.** Грузить из локальных snapshot-папок. Florence-2 — нативные чекпойнты `florence-community` (без `trust_remote_code`), в float16.
- **Драйвер и CUDA.** Проверить `nvidia-smi` на машине оценки (версия драйвера неизвестна). CUDA 13 требует драйвер ≥580, CUDA 12.x — ≥525. CUDA 13 убрала только архитектуры до Turing, так что T4 и GTX 1650 (sm_75) поддерживаются.
- **OpenCV.** В `requirements.txt` только `opencv-python==4.14.0.94`. `run_submission.py` импортирует `cv2`; `opencv-python-headless` из стартового kit удалить.
- **Всё пинить.** ultralytics, supervision и trackers выходят еженедельно или чаще.

---

## 3. Датасеты

**Общие правила.**
- **Запрещено:** видео той же камеры/перекрёстка, полученное другими путями. Не скрейпить YouTube, Telegram и городские камеры площадки. Перед обучением на краулинговых CCTV-наборах (ACCIDENT, TAD, CADP, HWID12) проверить, что ни один клип не показывает место соревнования; если показывает — исключить.
- **Время — в секундах.** Наши сэмплы 29.97 fps, датасеты — 4–50 fps, PDF говорит про 25 fps. Всё пересэмплировать по времени, fps не хардкодить.
- **Лицензии зеркал ненадёжны:** Kaggle/HF-перезаливы часто подписаны неверно (UA-DETRAC как CC0, TAD как MIT, UCF-Crime как CC0). В README всегда указывать лицензию оригинала.
- **Лучше прикреплять в Kaggle-ноутбуках**, чем качать локально: данные подключаются без скачивания.

### 3.1 Сводная таблица

| Датасет | Домен | Разметка | Размер | Лицензия | Наши классы | Доступность | Рекомендация |
|---|---|---|---|---|---|---|---|
| **ACCIDENT** (arXiv 2604.09819, 2026) | Неподвижные CCTV (+ синтетика CARLA с CCTV-ракурса) | Кадр удара (медиана первого контакта от 3–5 аннотаторов), место, тип столкновения (head-on, rear-end, t-bone, sideswipe, single-vehicle) + rollover. Синтетика: 2D-боксы, семантическая сегментация, ID, траектории, логи столкновений, 3D-состояние, LiDAR. **Конец аварии не размечен** | 2 027 реальных + 2 211 синтетических клипов; 20.16 GB (Kaggle API). Реальные обрезаны до ~30 с вокруг удара (1–114 с, медиана 26.8 с), 4–50 fps, 314p–3840p | **Конфликт:** Kaggle — CC BY-NC-SA 4.0; статья — аннотации CC BY 4.0, код и ассеты CARLA Apache-2.0. В репо GitHub нет LICENSE. Следовать строгому (NC-SA), цитировать статью | accident, Part B (0–5 с до удара — позитивы) | Kaggle `picekl/accident` (v9, 2026-04-10); `kaggle datasets download -d picekl/accident` или `dataset/download_dataset.sh` (bash, unzip, rsync, kaggle CLI → на Windows через WSL) | **основной** (#1) |
| **TAD** (WSAL, Lv et al., TIP 2021) | Дорожные камеры наблюдения (в основном Китай, часто низкое разрешение) | Метки уровня видео для train; подтипы в префиксах имён: 01_Accident, 02_IllegalTurn, 03_IllegalOccupation, 05_else, 06_PedestrianOnRoad, 07_RoadSpills, Normal (04 — предположительно Retrograde, не подтверждено). `label.zip` — возможно, временные метки теста (не подтверждено) | 500 видео (250 аномальных / 250 нормальных; 400 train / 100 test), в среднем 1 075 кадров, аномалия ~80 кадров. «25 hours» из README не сходится (~5 ч при 30 fps) — длительность не подтверждена. Зеркало Kaggle 13.44 GB (JPG-кадры) | Не указана (нет LICENSE); MIT на Kaggle — заявление загрузчика. Research, цитировать TIP 2021 | accident, illegal_turn / illegal_u_turn, stopped_vehicle, wrong_way, jaywalking, road_obstacle; Normal — hard negatives | Drive-папка (frames_part_1/2/3.zip, label.zip) или Kaggle `nikanvasei/traffic-anomaly-dataset-tad`; склейка кадров в mp4 — `stitch_tad_frames.py` из NVIDIA TAR или ffmpeg; fps не задокументирован (30 — допущение) | **основной** (#2) |
| **AI City Challenge 2021 Track 4** | Неподвижные камеры, хайвей, 410p, 30 fps | Train: `train-anomaly-results.txt` — `<video_id> <start> <end>` в секундах; аварии и заглохшие ТС | 100 train + 150 test видео по ~15 мин; zip 15.06 GB | Не указана; с января 2023 доступ к прошлым датасетам — через онлайн AI City Challenge Datasets Request Form (условия не подтверждены); research | stopped_vehicle, accident, контроль ложных срабатываний на длинных видео | Drive-zip жив (HTTP 200); сначала подать форму, потом `gdown 1gsR4pIEiWK7htPmIjiZjXetaa1X-T-Am`. Тестовые метки, вероятно, не публиковались (не подтверждено) | **основной** (#3) |
| **UA-DETRAC** | Неподвижная приподнятая камера (Китай, 2015) | Боксы и треки ТС, ignore-регионы, атрибуты перекрытия/погоды | 100 последовательностей, 140k+ кадров, 960x540 @25 fps (остальные цифры — 10 ч, 8 250 ТС, 1.21M боксов — не подтверждены). Kaggle-зеркала 11.51 GB (bratjay) / 9.89 GB (dtrnngc) | CC BY-NC-SA 3.0, «academic use only» (Wayback старого сайта). Метки зеркал (Unknown, CC0) неверны | Дообучение детектора ТС, оценка трекера (ID switches) | Официальные Google Drive ссылки на странице Dawei Du; или Kaggle-зеркала. Конвертировать XML → YOLO/COCO, закрасить ignore-регионы | **основной** (#4) |
| **D-Fire** | В основном уличные и лесные пожары | Боксы fire / smoke | 21 527 изображений (1 164 только огонь, 5 867 только дым, 4 658 огонь+дым, 9 838 негативов); 14 692 бокса огня, 11 865 дыма. Kaggle YOLO-версия 3.12 GB | CC0-1.0 | fire_smoke | GitHub `gaia-solutions-on-demand/DFireDataset` (старый `gaiasd/...` редиректит) | **основной** (#5) |
| **RAOD** (UnicomAI RAODBench) | Камеры ITS на хайвеях (Китай) | Попиксельные маски брошенных на дороге предметов | 557 видеопоследовательностей, 18 953 изображения; zip 5.41 GB | Лицензии нет. Research; цитировать IEEE Access 12:123985–123994 (2024), doi 10.1109/ACCESS.2024.3407955 | road_obstacle (маски → боксы связными компонентами) | `gdown 1WsaBYKtHT55_bdx0JW2vohxsXjFYMdh9` или Baidu (код 5tGb) | **основной** (#6) |
| **MIO-TCD Localization** | Тысячи реальных дорожных камер (Канада, США) | Боксы, 11 классов: articulated truck, bicycle, bus, car, motorcycle, motorized vehicle, non-motorized vehicle, pedestrian, pickup truck, single-unit truck, work van | 137 743 изображения, tar 3.74 GB (Classification: 648 959 кропов, 3.12 GB) | CC BY-NC-SA 4.0 | Детектор, включая пешеходов и двухколёсных (дополняет UA-DETRAC) | Прямая загрузка без регистрации; `gt_train.csv` → YOLO, взять 10–20k изображений. Разрешение кадров (не подтверждено); публичные метки, вероятно, только train (не подтверждено) | **основной** (#7) |
| **CADP** | CCTV с YouTube (оверлеи, дубликаты) | 1 416 сегментов аварий; 205 с полной пространственно-временной разметкой (JSON покрывает 226 видео, ключи — YouTube ID); боксы включают людей и двухколёсных | `segments.tar.gz` 10.3 GB + `annotations_CADP.json` 568 KB; в среднем 366 кадров (макс. 554) | Некоммерческое исследовательское использование, для другого — разрешение авторов. Код — GPL-3.0 (не копировать в репо) | accident, Part B (в среднем 3.69 с от начала клипа до аварии) | Главная Drive-папка жива; ссылка «Extracted Frames» — 404 (кадры есть в главной папке). Дедуплицировать с ACCIDENT/TAD по YouTube ID | **основной** (#8) |
| **COCO 2017** | Общие фото | Боксы, 80 классов (person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign, dog, horse, sheep, cow…) | train2017.zip 19.34 GB, аннотации 253 MB | Аннотации CC BY 4.0 (не перепроверено в этом прогоне); изображения — по-картиночные Flickr-лицензии, много NC/ND: CC BY-NC-SA 2.0, BY-NC 2.0, BY-NC-ND 2.0, BY 2.0, BY-SA 2.0, BY-ND 2.0, «No known copyright restrictions», «US Government Work». Изображения не распространять | Rehearsal против забывания (люди, двухколёсные, животные, светофоры); животные для road_obstacle | Прямая загрузка; только нужные классы через pycocotools или FiftyOne | **основной** |
| **Street Scene** (MERL, WACV 2020) | Одна неподвижная камера над двухполосной улицей (США), день | Покадровые боксы с ID трека на событие; 205 аномальных событий 17 типов в тесте: jaywalking 61, biker outside lane 42, loitering 36, dog on sidewalk 11, car outside lane 10, car u-turn 5, car illegally parked 5… | 46 train + 35 test последовательностей, 202 545 кадров 1280x720 @15 fps; zip 48.98 GB | CC BY-SA 4.0 (Zenodo) | Валидация правил: jaywalking, illegal_u_turn, stopped_vehicle / стоянка, движение вне полосы; точные start/end для калибровки конвенций | Zenodo 10870472, без регистрации: `wget -O StreetScene.zip 'https://zenodo.org/records/10870472/files/StreetScene.zip?download=1'` | запасной |
| **TADBench** (UnicomAI) | Хайвейные CCTV (Китай) | Метки уровня видео (без времени начала) | README: «344 videos … 277 positive … 127 negative» (277+127=404 — числа не сходятся); zip 4.54 GB; раскладка `{train,test}/<class>/<name>.mp4` | Лицензии нет; цитировать IEEE Access 13:2018–2033 (2025), doi 10.1109/ACCESS.2024.3522384 | accident (позитивы/негативы), кросс-датасетная проверка | `gdown 14GNlNcWLzN-sbzvmrMuSbAg_rZZ5yd26` или `python download_videos.py --only tad_bench` (NVIDIA TAR) | запасной |
| **SO-TAD** | CCTV (surveillance-oriented) | Гранулярность (время начала) не подтверждена | ~27.4 GB (so_tad.z01..z50 по 537 MB + so_tad.zip 506 MB); в TAR — 2 186 клипов | Лицензии нет; цитировать Neurocomputing 618 (2025) 129061 | accident, anticipation | `huggingface-cli download cccccxy/so-tad --repo-type dataset --local-dir so_tad`, затем `7z x so_tad.zip` (файлы в корне репо, не в `data/`) | запасной |
| **NVIDIA PhysicalAI-Traffic-Anomaly-Reasoning (TAR)** — AI City 2026 Track 3 | Агрегат CCTV (8 источников) | 44 040 псевдо-размеченных QA/CoT (Gemini 3.1 Pro + Gemma-4); `temporal_localization.json` — окна `{start,end}` в MM:SS (разрешение 1 с) **и для нормальных клипов** тоже — фильтровать по классу источника. Человеком проверен только тест | Train: 3 670 видео (~26.1 ч); аннотации ~0.06 GB; видео ~150 GB (в основном UCF-Crime). Клипы: so-tad 2 186, TAD-benchmark 366, UCF_Crimes 334, HTV 254, TAD 191, Vad-R1 131, Barbados 128, Accident-Bench 80 | Аннотации CC BY 4.0 («ready for commercial use»); видео не распространяются, у каждого источника своя лицензия | Загрузчик одной командой; шумные псевдометки; данные для VLM | `pip install huggingface_hub gdown kaggle requests`; `python download_videos.py --out ./videos --only tad tad_bench htv so_tad` (ключи: vad_r1, tad, accident_bench, so_tad, tad_bench, htv, ucf_crime, barbados); нужны Kaggle-ключи, 7z, ffmpeg | запасной |
| **UCF-Crime** (класс Road Accidents) | Уличное видеонаблюдение | Метки видео; временные — только для теста (zip 177 KB) | 1 900 видео, 128 ч, 13 классов; zip 102.96 GB (Kaggle `bypktt/ucf-crimes` — полные 104.9 GB); ~150 видео Road Accidents по статье (не подтверждено) | Не указана; research (CVPR 2018). CC0 на HF и «Unknown» на Kaggle — не авторитетно | accident (позитивы), негативы для аварий и огня; Arson/Explosion — контекст огня | Быстрее всего — прикрепить Kaggle-зеркало и читать только `Videos/RoadAccidents`. Зеркало UNCC недоступно | запасной |
| **TUMTraf-Accid3nD** | Неподвижные придорожные камеры + LiDAR, хайвей A9 (Мюнхен), реальные аварии | 2D/3D боксы, ID треков, маски, OpenLABEL, HD-карта; 6 классов; есть перевороты и загорания ТС | ~46.6 GB; 111 945 кадров при 25 Hz, 2 634 233 бокса | CC BY-NC-SA 4.0; dev-kit MIT | Проверка траекторных правил аварий; несколько реальных кадров горящих ТС для fire_smoke | Nextcloud-шара (без формы); конвертация dev-kit'ом | запасной |
| **FASDD** | Общие огонь/дым (+ БПЛА и ДЗЗ в полной версии) | Боксы fire / smoke | Полная версия 82.1 GB; CV-подмножество (зеркало HF) 3.41 GB, ~95k изображений (12.5k огонь, 23.3k дым, 20.1k оба, 39.1k пустых) | **CC BY-SA 4.0** (DataCite; «CC BY 4.0» на HF-зеркале — метка загрузчика, а CC BY 4.0 у Crossref относится к статье) | fire_smoke (разнообразие дыма, негативы) | `huggingface-cli download seawsurf/fire_smoke_dataset_fasdd_cv --repo-type dataset`, взять 10–20k; цитировать DOI 10.57760/sciencedb.j00104.00103 | запасной |
| **MEVA** (Kitware/IARPA) | Неподвижные наземные камеры, учебный полигон (постановочные действия) | Временные границы активностей: vehicle_makes_u_turn, vehicle_turns_left/right, vehicle_starts/stops/reverses, person_abandons_package и др. | 328 ч (516 GB) наземного видео + 4.6 ч (26 GB) БПЛА; аннотировано 120.2 ч train-уровня и 64 ч eval-уровня | CC BY 4.0 (можно публиковать производные веса с атрибуцией) | illegal_u_turn, illegal_turn, stopped_vehicle; немного для road_obstacle (брошенный предмет) | AWS Open Data без регистрации: `aws s3 ls --no-sign-request s3://mevadata-public-01/drops-123-r13/`; брать только нужные клипы по аннотациям из `meva-data-repo` | опционально |
| **SinD / SinD 2.0** | Дрон, вид сверху, 6 регулируемых перекрёстков в Китае | Траектории, состояния светофоров, HD-карты; семантика нарушений: въезд на красный/жёлтый, нарушение направления полосы, перестроение через сплошную, кандидаты wrong-way, вторжение VRU, конфликты MprTTC | Tianjin — 7 ч, 13 000+ участников; полный объём (не подтверждено); публичный пример `scenarios_sample.json` 603 KB | **Конфликт:** файл LICENSE — CC0 1.0, бейдж README — «Dataset: Non-commercial»; код Apache-2.0. Считать некоммерческим до ответа авторов | Настройка и юнит-тесты правил: red_light, stop_line, wrong_way, solid_line_crossing, jaywalking, failure_to_yield, пороги near_miss. Обучать им image-модели нельзя | Заявка **по e-mail** с учебного адреса (li-yw23@mails.tsinghua.edu.cn / hong_wang@tsinghua.edu.cn), тема `[Apply for SinD] name_country(region)_organization`; может занять дни | опционально |
| **AI City 2021 Track 1** (подсчёт по манёврам) | Неподвижные камеры перекрёстков | ROI и movement-of-interest (MOI) на камеру, счёт машин/грузовиков | 31 клип (~9 ч), 20 ракурсов, «960p or better», в основном 10 fps; zip 5.83 GB | Не указана; через Datasets Request Form; research | Шаблон для конфигурации допустимых манёвров (illegal_turn, illegal_u_turn, wrong_way) | Форма, затем `gdown 1xGaOlCwz7Zn7SbDZiWsfG0m2UGs7eqpO` | опционально |
| **Open Images V7** | Общие фото | Боксы, 600 классов (train 1 743 042 изображения / 14 610 229 боксов) | По классам | Аннотации CC BY 4.0; изображения указаны как CC BY 2.0, но Google не гарантирует лицензию каждой картинки | road_obstacle: Cattle, Goat, Horse, Dog, Sheep, Deer, Animal, Box, Tire, Barrel, Cart, Wheel. **Класса «Traffic cone» нет** | `fiftyone.zoo.load_zoo_dataset('open-images-v7', split='train', label_types=['detections'], classes=[...], max_samples=...)` | опционально |
| **MITS** (UnicomAI) | Кадры видеонаблюдения | ~5M инструкций VQA: пробки, разливы, погода, стройка, фейерверки/дым, аварии | 170 400 изображений; 74.4 GB | Apache-2.0 (метаданные ModelScope); лицензии дообученных чекпойнтов неясны | Только если делаем VLM-верификатор | ModelScope (`modelscope` CLI/SDK) | опционально |
| **CCD** (Car Crash Dataset) | Dashcam (ego) | Покадровые бинарные метки (`Crash-1500.txt`), VGG-16 признаки | 1 500 аварий + 3 000 нормальных (из BDD100K), 50 кадров @10 fps; Kaggle 8.59 GB | Репо MIT (код и аннотации); видео — YouTube и BDD100K (лицензия BDD) | Предобучение Part B | Drive или Kaggle `asefjamilajwad/car-crash-dataset-ccd` | опционально |
| **BDD100K** | Dashcam (ego) | Боксы (светофоры с цветом), MOT-подмножество | 100K видео по 40 с; 100K изображений с боксами | UC Regents: образование, исследования, некоммерческие цели | Примеры светофоров и двухколёсных | Регистрация на `http://bdd-data.berkeley.edu/` (https и зеркала 2026-09-25 не отвечали) | опционально |
| **MOT17 / MOT20** | Пешеходы, в основном статичные/приподнятые камеры | MOT-треки | MOT20.zip 5.03 GB (+ метки 13.9 MB), 8 последовательностей 1920x1080 @25 fps; MOT17.zip 5.86 GB | Не указана на страницах (обычно цитируют CC BY-NC-SA 3.0 — не подтверждено) | Оценка трекера на плотных переходах | Загрузки живы; онлайн-оценка закрыта (лидерборд в архиве с 2026-04-16) | опционально |
| **Barbados Traffic Analysis Challenge** (Zindi) | 4 камеры на кольце Norman Niles | Уровни загруженности (4 класса), ~1-минутные сегменты | Kaggle-перекодировка 126.46 GB; полный набор >500 GB | Лицензии нет; правила Zindi, **ручная разметка запрещена** правилами соревнования; CC BY 4.0 на Kaggle — не авторитетно | congestion | Только при подтверждённой лицензии | избегать |
| **Objects365** | Общие фото | 365 классов, ~2M изображений, 30M боксов | Сотни GB (не подтверждено) | «available for the academic purpose only»; изображения не распространять | Только как предобучение чужих чекпойнтов — проверять их лицензию | Не качать | избегать |
| **Accident-Bench** (M4R) | Смешанные (суша, вода, воздух) VQA | ~19 000 QA с множественным выбором | ~42.6 GB | Лицензии нет | Нет временных меток для нашей задачи | — | избегать |
| **TT100K** | Дорожные **знаки** | — | 100K изображений (не подтверждено) | (не подтверждено) | Не наша задача (знаки, не светофоры) | — | избегать |

**Также проверены (без изменений), кратко:**

| Датасет | Суть | Лицензия / условия | Для чего нам |
|---|---|---|---|
| Highway Traffic Videos (HTV) | 92 MB, Kaggle; первый кадр пропускать; 254 клипа совпадают с TAR | Kaggle CC0; WSDOT courtesy | congestion |
| HWID12 | 12.03 GB | Только research/education; номера размывать | Доп. CCTV (проверить на кадры площадки) |
| TU-DAT | 49 позитивов по ~11 с @30 fps (16 170 кадров), ~8 800 кадров негативов, папка Rash-Driving | Лицензии нет | accident |
| S2TLD | 5 786 изображений, 14 130 светофоров; HF 1.43 GB | Репо MIT, HF Apache-2.0 | Предобучение классификатора светофора |
| LISA Traffic Light | 5.16 GB (Kaggle) | CC BY-NC-SA 4.0 | Кропы светофоров (dashcam) |
| Bosch Small Traffic Lights | Zenodo 37.66 GB, `dataset_train_rgb.zip` 6.18 GB | Некоммерческая лицензия (PDF) | Кропы светофоров (dashcam) |
| Pyro-SDIS | HF `pyronear/pyro-sdis` | Apache-2.0 | fire_smoke (дым) |
| Roboflow fire `personal-bodxv/fire-detection-sejra-fognw` | 8 939 изображений | CC BY 4.0 | fire_smoke (сравнение) |
| DFS | 9 462 изображения, VOC, fire/smoke/other; ссылка OneDrive не проверялась | Лицензии нет | fire_smoke |
| Simuletic | 220 синтетических изображений, текстовые метки JSONL, без боксов | CC BY-NC 4.0 | — |
| Kaggle/Roboflow кадры аварий | `ckay16` 0.26 GB («Open Database, Contents © Original Authors»); `justjuu` 2 763 изображения (2 122+316+325), тег CC0 | См. слева | Только эксперименты |
| VisDrone | 288 клипов / 261 908 кадров / 10 209 изображений, вид с дрона | Лицензии нет | Не для наших image-моделей |
| CrowdHuman | HF 14.2 GB; 15 000 / 4 370 / 5 000 изображений, 470K инстансов | CC BY-NC 4.0 + no-redistribution | В AGPL-репо не использовать |
| CityPersons / Cityscapes | Городские сцены (ego) | Некоммерческие, без распространения; обученные модели — «abstract representations» | Не нужны |
| MM-AU | HF 525.8 GB, «ONLY free for academic use» | CC BY-NC 4.0 | Dashcam, максимум предобучение |
| DAD | Оригинальный Drive — 401; Kaggle-зеркало 2.83 GB, 1 750 клипов | Права у авторов | Dashcam, максимум предобучение |
| PIE / JAAD | ~74 GB / 346 клипов (~82k кадров) | MIT / MIT | Пешеходы с ego-камеры, не наш ракурс |
| TUMTraf Intersection / VideoQA | Нужна регистрация | (не подтверждено) | Опционально |
| SegmentMeIfYouCan | — | — | избегать |

### 3.2 Топ-8 для нашего проекта и план данных

1. **ACCIDENT** — реальные CCTV-аварии с кадром удара, местом и типом + синтетика CARLA с треками. Модель аварий и Part B. Лицензия — конфликт (Kaggle CC BY-NC-SA 4.0 против CC BY 4.0 в статье) → указываем строже. Официальный IID-сплит маленький (507 train / 1 520 test реальных), для нас можно учить на всём. Конец аварии не размечен — смещение конца калибровать на нашем dev-сете.
2. **TAD (WSAL)** — 500 видео с дорожных камер: аварии, незаконные повороты, незаконная стоянка, (предположительно) встречное движение, пешеходы на дороге, разливы на дороге. Длительность «25 h» сомнительна.
3. **AI City 2021 Track 4** — заглохшие ТС и аварии, 100+150 видео по ~15 мин, start/end в секундах. 15.06 GB, **нужна Datasets Request Form** (фраза «без формы» не подтвердилась).
4. **UA-DETRAC** — приподнятая неподвижная камера, треки ТС. CC BY-NC-SA 3.0.
5. **D-Fire** — 21.5k изображений огня/дыма, CC0.
6. **RAOD** — 557 последовательностей брошенных на дороге предметов с масками, 5.41 GB.
7. **MIO-TCD Localization** — 137k изображений с дорожных камер, 11 классов с пешеходами и двухколёсными, 3.74 GB прямой загрузкой. CC BY-NC-SA 4.0.
8. **CADP** — CCTV-аварии с пространственно-временными треклетами для прогнозирования (Part B). Реальные файлы: `segments.tar.gz` 10.3 GB + `annotations_CADP.json`; ссылка на кадры мертва, главная папка жива.

Плюс **COCO 2017** (rehearsal) и **Street Scene** (валидация правил на неподвижной уличной камере). Для `near_miss` публичного CCTV-датасета **не нашлось** — выводим из траекторий (TTC/PET + замедление) и настраиваем на своих метках (конфликтные траектории SinD помогают выставить пороги).

**Бэкапы:** TADBench, SO-TAD, UCF-Crime (Road Accidents); загрузчик NVIDIA TAR (TAD, TADBench, SO-TAD, HTV, UCF-Crime одной командой); TUMTraf-Accid3nD (реальные придорожные аварии с треками); FASDD (второй набор огня/дыма); Highway Traffic Videos (пробки); SinD 2.0 (нарушения с сигналами — для правил).

#### День 0 (параллельно)

1. Прокси-видео с тем же числом кадров, проверка ffprobe (§1.3).
2. Геометрия сцены: проезжая часть, стоп-линии, переходы, разделители и сплошные, допустимые манёвры по подходам, ROI светофоров (§1.4).
3. Загрузки — в Kaggle прикрепить сразу ACCIDENT, TAD (зеркало), UA-DETRAC (зеркало), UCF-Crime (зеркало, если нужно); локально/в облако — по порядку:

| Порядок | Датасет | Размер |
|---|---|---|
| 1 | D-Fire | ~3.1 GB (Kaggle YOLO-версия) |
| 2 | MIO-TCD Localization | 3.74 GB |
| 3 | RAOD | 5.41 GB |
| 4 | UA-DETRAC | ~9.9–11.5 GB (зеркала) |
| 5 | TAD (Kaggle) | 13.44 GB |
| 6 | AI City 2021 Track 4 (после формы) | 15.06 GB |
| 7 | ACCIDENT (лучше прикрепить в Kaggle) | 20.16 GB |
| 8 | CADP | 10.3 GB + 568 KB аннотаций |
| — | Остальное | только если появится пробел |

4. **Сразу подать заявки:** SinD (e-mail, может занять дни) и AI City Challenge Datasets Request Form.

#### Своя разметка — самые ценные данные

- **События:** разметить все сэмплы в формате организаторов; минимум одно видео — двойная разметка и замер согласия границ; одностраничная конвенция границ (§1.2); **одно видео целиком — hold-out** для теста.
- **Боксы:**
  1. Сильный «учитель» на 4K-тайлах + трекер для связи боксов.
  2. Человеком поправить ~300–600 ключевых кадров (X-AnyLabeling или CVAT): ТС, двухколёсные, пешеходы, животные/препятствия.
  3. Уверенные боксы ещё на ~5–10k кадрах принять как псевдометки.
  4. С hold-out видео псевдометки для обучения **не** брать.

#### Обучающие смеси

- **Основной детектор:** ~40% наших кадров (с оверсэмплингом — единственные данные из домена) + ~40% данных неподвижных дорожных камер (подмножества UA-DETRAC и MIO-TCD, 10–20k изображений) + ~20% подмножества COCO (люди, двухколёсные, животные, светофоры) как rehearsal.
- **fire_smoke:** отдельная маленькая модель на D-Fire (+ подмножество FASDD / Pyro-SDIS) + hard negatives с наших видео: стоп-сигналы, красные машины, блики солнца, пыль, выхлоп.
- **road_obstacle:** вычитание фона («новый статичный объект внутри полигона дороги, не трек ТС, держится ≥N с») + детектор на RAOD и животных COCO / Open Images.
- **accident / near_miss:** классификатор/локализатор аварий на ACCIDENT (реальные + синтетика), TAD, авариях AIC21-T4, CADP, TADBench; пороги калибровать на dev-сете. near_miss — из траекторий.
- **Part B:** позитивные кадры — 0–5 с до удара из ACCIDENT, AIC21-T4, TAD, CADP; остальное — негативы; модель строго каузальная. Dashcam-наборы (CCD, DAD, MM-AU) — максимум предобучение.
- **Классы-правила** (red_light, stop_line, wrong_way, illegal_turn, illegal_u_turn, solid_line_crossing, stopped_vehicle, jaywalking, failure_to_yield, congestion): правила по траекториям + геометрия. Публичные данные — только для проверки: подтипы TAD, заглохшие ТС AIC21-T4, пробки HTV, нарушения SinD, события Street Scene, манёвры MEVA. Состояние светофора — ROI + маленький классификатор на кропах из наших видео (предобучение на кропах S2TLD / LISA).

#### Конвенции времени в датасетах (разные!)

- ACCIDENT: медиана кадра первого контакта; конца нет.
- AIC21-T4: start/end, исходная метрика засчитывает детекцию в пределах 10 с от начала — границы грубее, чем нужно для tIoU 0.7.
- TAD: покадровые маски теста; train — только уровень видео.
- NVIDIA TAR: окна MM:SS (1 с), есть и у нормальных клипов.
- Street Scene: кадры @15 fps.
- Всё переводить в секунды и калибровать смещения начала/конца на нашем dev-сете.

#### Предупреждения

- **Мёртвые/изменившиеся ссылки (проверено 2026-09-25):** ссылка «кадры» CADP — 404 (кадры есть в главной папке); оригинальный Drive DAD — 401 (есть Kaggle-зеркало); старый сайт UA-DETRAC редиректит (официальные Drive-ссылки — на странице Dawei Du, плюс Kaggle); bdd100k.com и doc.bdd100k.com не отвечали; зеркало UCF-Crime на UNCC недоступно; AI City теперь требует форму запроса.
- **Barbados (Zindi)** — данные соревнования без явной лицензии; не брать без подтверждения.
- **Приватность:** в краулинговых CCTV есть номера и лица — размывать в любом демо на публичном сайте (HWID12 требует этого прямо).
- **Бюджет:** веса ≤5 GB, инференс — в пределах T4; обучение — на Kaggle/Colab.

### 3.3 Таблица лицензий для README

**Датасеты** (указывать лицензию **оригинала**, не зеркала):

| Датасет | Лицензия (как проверено) | Наше использование |
|---|---|---|
| ACCIDENT | Kaggle: CC BY-NC-SA 4.0; статья: аннотации CC BY 4.0, код и ассеты CARLA Apache-2.0 → применяем CC BY-NC-SA 4.0; цитировать arXiv 2604.09819 | accident / Part B |
| TAD (WSAL) | Не указана; research (MIT на Kaggle-зеркале не авторитетно); цитировать TIP 2021 | Аномалии, подтипы |
| AI City Challenge 2021 Track 4 / Track 1 | Не указана; доступ через Datasets Request Form (условия не подтверждены); research | stopped_vehicle / accident; геометрия манёвров |
| UA-DETRAC | CC BY-NC-SA 3.0, academic use only | Детектор / трекер |
| MIO-TCD | CC BY-NC-SA 4.0 | Детектор |
| CADP | Некоммерческое исследовательское использование (условия авторов); код GPL-3.0 | accident / Part B |
| COCO 2017 | Аннотации CC BY 4.0 (не перепроверено); изображения — по-картиночные Flickr-лицензии CC 2.0, **включая NC/ND** | Предобучение / rehearsal |
| D-Fire | CC0-1.0 | fire_smoke |
| Pyro-SDIS | Apache-2.0 | fire_smoke |
| FASDD | **CC BY-SA 4.0** (DataCite) | fire_smoke |
| RAOD | Не указана; research; IEEE Access 2024, doi 10.1109/ACCESS.2024.3407955 | road_obstacle |
| TADBench | Не указана; research; IEEE Access 2025, doi 10.1109/ACCESS.2024.3522384 | accident |
| SO-TAD | Не указана; research; Neurocomputing 618 (2025) 129061 | accident |
| TU-DAT | Не указана | accident |
| UCF-Crime | Не указана; research (CVPR 2018) | Позитивы/негативы аварий |
| TUMTraf-Accid3nD | CC BY-NC-SA 4.0 (dev-kit MIT) | accident / треки |
| Street Scene | CC BY-SA 4.0 | Валидация правил |
| MEVA | CC BY 4.0 | Манёвры (u-turn, повороты, остановки) |
| Highway Traffic Videos | CC0 (Kaggle) / WSDOT courtesy | congestion |
| SinD | **Конфликт:** LICENSE-файл CC0 1.0 против бейджа «Dataset: Non-commercial»; код Apache-2.0 → считаем некоммерческим | Настройка правил |
| S2TLD | MIT (репо) / Apache-2.0 (HF) | Светофор |
| LISA Traffic Light | CC BY-NC-SA 4.0 | Светофор |
| Bosch Small Traffic Lights | Некоммерческая лицензия | Светофор |
| Open Images V7 | Аннотации CC BY 4.0; изображения CC BY 2.0 без гарантий | Животные / предметы |
| NVIDIA TAR | Аннотации CC BY 4.0; у видео — лицензии источников | Загрузчик / VLM |
| MITS | Apache-2.0 (ModelScope) | VLM (если будет) |
| CCD | MIT (код и аннотации); видео — YouTube / BDD100K | Предобучение Part B |
| BDD100K | UC Regents: educational, research, not-for-profit | Светофоры / двухколёсные |

**Модели и код** (что реально попадает в репо или используется при обучении):

| Компонент | Лицензия |
|---|---|
| Ultralytics (YOLO26 / YOLO11 / YOLOE-26, трекеры) и обученные ими веса | AGPL-3.0 |
| `rabahdev/fire-smoke-yolov8n` | AGPL-3.0 |
| supervision | MIT |
| OpenCV (`opencv-python`) | Apache-2.0 |
| PyAV | BSD-3-Clause (в колёсах — GPL-сборка FFmpeg) |
| VideoMAE V2 (код / distilled веса) | MIT / тег apache-2.0 (данные K710) |
| V-JEPA 2 / 2.1 | MIT |
| Autoware traffic_light_classifier | Apache-2.0 |
| MM-Grounding-DINO / OWLv2 / Grounding DINO | Apache-2.0 |
| RF-DETR N/S/M/L, RF-DETR-Seg | Apache-2.0 |
| RAFT (torchvision) | BSD-3-Clause |
| Two-Dimensional-TTC, SSMsOnPlane, ActionFormer, TriDet | MIT |
| Qwen3-VL-2B, MiniCPM-V 4.6, Qwen3.5 | Apache-2.0 |
| Cosmos-Reason2-2B, Cosmos-Embed1 | NVIDIA Open Model License |

**Что ещё написать в README:**
- Датасеты в репо **не** распространяются, только скрипты загрузки.
- Если используется Ultralytics — код репо под AGPL-3.0.
- Веса, обученные на NC/SA-данных, наследуют эти условия: выпускать «for research / non-commercial use, CC BY-NC-SA 4.0-compatible».
- **Лицензионная развилка (решить командой, при сомнении спросить организаторов; это не юридическая консультация).** Ultralytics считает обученные модели AGPL-3.0, а AGPL не допускает дополнительных ограничений вроде «только некоммерческое». Чтобы не смешивать, вариантов два: (a) учить Ultralytics-детектор только на данных без NC (наши кадры, аннотации COCO, D-Fire, Pyro-SDIS, MEVA, Street Scene с SA), а NC-наборы (UA-DETRAC, MIO-TCD, ACCIDENT…) использовать для моделей вне Ultralytics (VideoMAEv2 — MIT-код); (b) пермиссивный путь RF-DETR (Apache-2.0) и явная пометка NC-SA на веса. SA-наборы (FASDD, Street Scene) требуют share-alike для производных данных; распространяется ли это на веса — юридически не решено.
- ND / no-redistribution (Cityscapes, CrowdHuman, возможно TUMTraf): данные и изменённые копии никогда не публиковать.

---

## 4. Ссылки

### 4.1 Инструменты разметки и утилиты

- Label Studio: https://github.com/HumanSignal/label-studio · https://labelstud.io/tags/timelinelabels · https://labelstud.io/tags/video · ML-бэкенды: https://github.com/HumanSignal/label-studio-ml-backend
- X-AnyLabeling: https://github.com/CVHub520/X-AnyLabeling
- labelme: https://github.com/wkentaro/labelme
- CVAT: https://github.com/cvat-ai/cvat · https://docs.cvat.ai · https://github.com/cvat-ai/cvat-models
- CVAT Online: https://app.cvat.ai · https://www.cvat.ai/pricing/cvat-online
- ELAN: https://archive.mpi.nl/tla/elan · https://archive.mpi.nl/tla/elan/download · https://www.mpi.nl/tools/elan/docs/manual/index.html
- BORIS: https://www.boris.unito.it/ · https://github.com/olivierfriard/BORIS · Windows Portable: boris.unito.it/download_win/
- VIA 3: https://www.robots.ox.ac.uk/~vgg/software/via/ · zip: https://www.robots.ox.ac.uk/~vgg/software/via/downloads/via3/via-3.0.13.zip
- FFmpeg: https://ffmpeg.org · drawtext: https://ffmpeg.org/ffmpeg-filters.html#drawtext-1
- mpv: https://github.com/mpv-player/mpv · https://mpv.io/installation/
- uv: https://github.com/astral-sh/uv
- FiftyOne: https://github.com/voxel51/fiftyone · https://docs.voxel51.com
- Datumaro: https://github.com/open-edge-platform/datumaro
- opencv-python: https://pypi.org/project/opencv-python/ · https://github.com/opencv/opencv-python · миграция 4→5: https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration
- PyAV: https://pypi.org/project/av/
- makesense.ai: https://www.makesense.ai · https://github.com/SkalskiP/make-sense
- DarkLabel: https://github.com/darkpgmr/DarkLabel
- Избегать (для справки): Supervisely https://supervisely.com/pricing/ · https://docs.supervisely.com · https://github.com/supervisely/supervisely; Roboflow https://roboflow.com/pricing; Encord https://encord.com/pricing/; V7 Darwin https://www.v7darwin.com/pricing; Dataloop https://dataloop.ai; Xtreme1 https://github.com/xtreme1-io/xtreme1; ANVIL http://www.anvil-software.de/; Kinovea https://www.kinovea.org/download.html; AnyLabeling https://github.com/vietanhdev/anylabeling

### 4.2 Детекторы, трекеры, восприятие

- YOLO26: https://docs.ultralytics.com/models/yolo26/ · YOLO11: https://docs.ultralytics.com/models/yolo11/ · трекинг: https://docs.ultralytics.com/modes/track/
- Веса Ultralytics: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt (и `yolo26s.pt`, `yolo26m-seg.pt`, `yoloe-26l-seg.pt`, `yoloe-26s-seg.pt`, `mobileclip2_b.ts` в том же релизе) · https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt · https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf
- YOLOv10: https://github.com/THU-MIG/yolov10
- YOLOX: https://github.com/Megvii-BaseDetection/YOLOX
- RT-DETRv4: https://github.com/RT-DETRs/RT-DETRv4
- DEIM: https://github.com/Intellindust-AI-Lab/DEIM
- D-FINE (веса): https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_obj2coco.pth
- RF-DETR: https://github.com/roboflow/rf-detr · https://rfdetr.roboflow.com/latest/ · веса: https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth · https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth · https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth
- BoxMOT: https://github.com/mikel-brostrom/boxmot
- ByteTrack: https://github.com/FoundationVision/ByteTrack · BoT-SORT: https://github.com/NirAharon/BoT-SORT · OC-SORT: https://github.com/noahcao/OC_SORT · Deep OC-SORT: https://github.com/GerardMaggiolino/Deep-OC-SORT
- Autoware traffic_light_classifier: https://huggingface.co/AutowareFoundation/traffic_light_classifier · файл: https://huggingface.co/AutowareFoundation/traffic_light_classifier/resolve/main/traffic_light_classifier_mobilenetv2_batch_1.onnx
- Огонь/дым: https://huggingface.co/rabahdev/fire-smoke-yolov8n (веса: https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt)
- MM-Grounding-DINO tiny: https://huggingface.co/openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
- OWLv2: HF `google/owlv2-base-patch16-ensemble`
- SAM 2: https://github.com/facebookresearch/sam2 · веса tiny: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
- SAM 3: https://github.com/facebookresearch/sam3
- RAFT small (torchvision): https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth
- SEA-RAFT: https://github.com/princeton-vl/SEA-RAFT
- ONNX Runtime GPU: https://pypi.org/project/onnxruntime-gpu/
- PyNvVideoCodec: https://pypi.org/project/PyNvVideoCodec/ · TorchCodec: https://pypi.org/project/torchcodec/
- Индекс колёс PyTorch cu126: https://download.pytorch.org/whl/cu126

### 4.3 Модели событий, VLM, TTC

- VideoMAE V2: https://huggingface.co/OpenGVLab/VideoMAE2 · https://github.com/OpenGVLab/VideoMAEv2
- VideoMAE v1 (NC): https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics
- V-JEPA 2 / 2.1: https://github.com/facebookresearch/vjepa2 · https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2
- PyTorchVideo (X3D): https://github.com/facebookresearch/pytorchvideo
- UniFormerV2: https://github.com/OpenGVLab/UniFormerV2 · https://huggingface.co/Andy1621/uniformerv2
- InternVideo2: https://huggingface.co/OpenGVLab/InternVideo2_distillation_models · https://github.com/OpenGVLab/InternVideo · https://huggingface.co/OpenGVLab/InternVideo2_CLIP_S
- TimeSformer (NC): https://huggingface.co/facebook/timesformer-base-finetuned-k400
- MoViNet: https://github.com/Atze00/MoViNet-pytorch · https://huggingface.co/kfkas/movinet-a0-stream-pytorch · https://huggingface.co/litert-community/MoViNet-A0-Stream-LiteRT
- TimeSouth: https://github.com/TimeSouth/zero-shot-taa-solution · https://huggingface.co/TimeSouth/zero-shot-taa-2nd-place
- Accident-детекторы (избегать): https://huggingface.co/hilmantm/detr-traffic-accident-detection · https://huggingface.co/dri11heaD/rtdetr-vehicle-accident-detection
- Community-чекпойнты (избегать): https://huggingface.co/esmaelehab/videomae-base-finetuned-kinetics-finetuned-CCTV-Accident-3s-2 · https://huggingface.co/OPear/videomae-large-finetuned-UCF-Crime · https://huggingface.co/jatinmehra/Accident-Detection-using-Dashcam
- Cosmos-Embed1 anomaly: https://huggingface.co/nvidia/Cosmos-Embed1-448p-anomaly-detection
- VAD (избегать): https://github.com/nwpu-zxr/VadCLIP · https://github.com/tianyu0207/RTFM · https://github.com/henrryzh1/UR-DMU
- Qwen3-VL-2B: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct · Qwen3.5-2B: https://huggingface.co/Qwen/Qwen3.5-2B
- MiniCPM-V 4.6: https://huggingface.co/openbmb/MiniCPM-V-4.6 · MiniCPM-V 4.5: https://huggingface.co/openbmb/MiniCPM-V-4_5
- Cosmos-Reason2-2B: https://huggingface.co/nvidia/Cosmos-Reason2-2B
- Phi-4-multimodal: https://huggingface.co/microsoft/Phi-4-multimodal-instruct
- Zero-shot энкодеры: https://huggingface.co/google/siglip2-base-patch16-224 · https://huggingface.co/facebook/PE-Core-S16-384 · https://huggingface.co/microsoft/xclip-base-patch32
- vLLM (установка GPU): https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html
- Two-Dimensional-TTC: https://github.com/Yiru-Jiao/Two-Dimensional-Time-To-Collision (SSMsOnPlane — репо того же автора)
- Emergency Index: https://github.com/AutoChengh/EmergencyIndex · MEI: https://github.com/AutoChengh/MEI
- Traffic Intelligence: https://bitbucket.org/Nicolas/trafficintelligence
- LibSSM (избегать): https://github.com/ObliviateRickLin/LibSSM
- ActionFormer: https://github.com/happyharrycn/actionformer_release · TriDet: https://github.com/dingfengshi/TriDet · OpenTAD: https://github.com/sming256/OpenTAD

### 4.4 Датасеты

- ACCIDENT: https://accidentbench.github.io · https://www.kaggle.com/datasets/picekl/accident · https://github.com/accidentbench/ACCIDENT · https://arxiv.org/abs/2604.09819
- TAD (WSAL): https://github.com/ktr-hubrt/WSAL · https://drive.google.com/open?id=1cofMJGglil4vddrq_unuy7EEhthMYtuq · https://www.kaggle.com/datasets/nikanvasei/traffic-anomaly-dataset-tad · https://arxiv.org/abs/2008.08944
- AI City 2021: https://www.aicitychallenge.org/2021-data-and-evaluation/ · Track 4: https://www.aicitychallenge.org/2021-track4-download/ · https://drive.google.com/file/d/1gsR4pIEiWK7htPmIjiZjXetaa1X-T-Am/view · Track 1: https://www.aicitychallenge.org/2021-track1-download/ · https://drive.google.com/file/d/1xGaOlCwz7Zn7SbDZiWsfG0m2UGs7eqpO/view · доступ: https://www.aicitychallenge.org/2021-data-access-instructions/
- UA-DETRAC: https://sites.google.com/view/daweidu/projects/ua-detrac · https://arxiv.org/abs/1511.04136 · https://www.kaggle.com/datasets/bratjay/ua-detrac-orig · https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset
- RAOD: https://github.com/UnicomAI/UnicomBenchmark/tree/main/RAODBench · https://drive.google.com/file/d/1WsaBYKtHT55_bdx0JW2vohxsXjFYMdh9/view · https://pan.baidu.com/s/1MdjOxZ2TQ-5PX_cB6PJQYg (код 5tGb)
- MIO-TCD: https://tcd.miovision.com/challenge/dataset.html · https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar
- CADP: https://ankitshah009.github.io/accident_forecasting_traffic_camera · https://drive.google.com/drive/folders/1ikepACwlFsVeaYSRWOQGDQXh1Hy5gWS6 · https://filedn.com/lQfYylUHSWEYAyORS7PdIKH/annotations_CADP.json · код: https://github.com/ankitshah009/CarCrash_forecasting_and_detection
- COCO 2017: https://cocodataset.org · http://images.cocodataset.org/zips/train2017.zip
- D-Fire: https://github.com/gaia-solutions-on-demand/DFireDataset
- Pyro-SDIS: https://huggingface.co/datasets/pyronear/pyro-sdis
- FASDD: https://doi.org/10.57760/sciencedb.j00104.00103 · https://www.scidb.cn/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda · https://huggingface.co/datasets/seawsurf/fire_smoke_dataset_fasdd_cv · https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2347922
- TADBench: https://github.com/UnicomAI/UnicomBenchmark/tree/main/TADBench · https://drive.google.com/file/d/14GNlNcWLzN-sbzvmrMuSbAg_rZZ5yd26/view · https://pan.baidu.com/s/1X8xRJWZ5izXuyUgGbGppjw (код gi9f) · https://arxiv.org/abs/2209.12386
- SO-TAD: https://huggingface.co/datasets/cccccxy/so-tad · https://www.sciencedirect.com/science/article/abs/pii/S0925231224018320
- NVIDIA TAR: https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning · https://www.aicitychallenge.org/2026-track3/
- UCF-Crime: https://www.crcv.ucf.edu/projects/real-world/ · https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip · https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0 · https://www.crcv.ucf.edu/projects/real-world/Temporal_Anomaly_Annotation_For_Testing_Videos.zip
- TUMTraf-Accid3nD: https://accident-dataset.github.io · https://nx21496.your-storageshare.de/s/Zd4rwsMwwXMaXiK · https://arxiv.org/abs/2503.12095 · https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit
- Street Scene: https://zenodo.org/records/10870472 · https://www.merl.com/research/downloads/StreetScene · https://arxiv.org/abs/1902.05872
- MEVA: https://mevadata.org/ · https://mevadata-public-01.s3.amazonaws.com/ · https://gitlab.kitware.com/meva/meva-data-repo · https://openaccess.thecvf.com/content/WACV2021/html/Corona_MEVA_A_Large-Scale_Multiview_Multimodal_Video_Dataset_for_Activity_Detection_WACV_2021_paper.html
- SinD: https://github.com/SOTIF-AVLab/SinD · https://arxiv.org/abs/2607.16943 · https://arxiv.org/abs/2209.02297
- Open Images V7: https://storage.googleapis.com/openimages/web/index.html · https://storage.googleapis.com/openimages/web/factsfigures_v7.html
- MITS: https://github.com/UnicomAI/UnicomBenchmark/tree/main/Multimodal-Intelligent-Traffic-Surveillance · https://www.modelscope.cn/datasets/zhaokaikai/Multimodal_Intelligent_Traffic_Surveillance
- CCD: https://github.com/Cogito2012/CarCrashDataset · https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F · https://www.kaggle.com/datasets/asefjamilajwad/car-crash-dataset-ccd
- BDD100K: http://bdd-data.berkeley.edu/ · https://github.com/bdd100k/bdd100k
- MOT: https://motchallenge.net/data/MOT17/ · https://motchallenge.net/data/MOT20/
- Barbados (избегать): https://zindi.world/competitions/barbados-traffic-analysis-challenge/data · https://www.kaggle.com/datasets/kipngetichv/reencoded-barbados-traffic
- Objects365 (избегать): https://www.objects365.org/overview.html · https://www.objects365.org/download.html
- Accident-Bench (избегать): https://huggingface.co/datasets/Open-Space-Reasoning/AccidentBench · https://accident-bench.github.io/ · https://arxiv.org/abs/2509.26636
- TT100K (избегать): https://cg.cs.tsinghua.edu.cn/traffic-sign/

### 4.5 Статьи, на которые опирается план

- ACCIDENT: https://arxiv.org/abs/2604.09819
- SynCrash (VideoMAEv2 на CARLA, ACCIDENT@CVPR 2026): https://arxiv.org/abs/2608.29759
- TimeSouth (CVPR 2026 AUTOPILOT, 2-е место): https://arxiv.org/abs/2606.09542
- Qwen3-VL-32B + YOLO11x + BoT-SORT на реальных CCTV: https://arxiv.org/abs/2608.08867
- Zero-shot CLIP-пайплайн ACCIDENT@CVPR 2026: https://arxiv.org/abs/2604.09685
- RF-DETR: https://arxiv.org/abs/2511.09554 · FastTracker: https://arxiv.org/abs/2508.14370
- CADP: https://arxiv.org/abs/1809.05782

---

*Каталог собран по проверенному корпусу 2026-09-25. Поля с пометкой «(не подтверждено)» и все оценки скорости без ссылки на таблицу авторов перепроверить до того, как на них опираться. Если организаторы опубликуют определения классов, пример GT или формат тестовых видео — они важнее этого документа.*
