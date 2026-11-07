# Контекст проекта для Qwen Coder

## О проекте

Магистерский проект УлГТУ, студент Гордеев И.В. (группа ИИБАмд-11), научный руководитель Мошкин В.С.

Тема диссертации: "Разработка метода оценки качества сгенерированных с помощью LLM UML-диаграмм".

Текущий модуль — извлечение UML class diagrams из изображений. Это второй из трёх модулей общей системы:
1. Модуль извлечения из текстовых требований (NLP)
2. **Модуль извлечения из изображений UML (CV+OCR)** — текущий фокус
3. База знаний БТ/ФТ/НФТ + алгоритм оценки качества

## Цель модуля

Принимает PNG-изображение UML class diagram (экспорт или скриншот) → возвращает структурированный JSON с классами, атрибутами, методами и связями.

JSON должен быть совместим с будущим модулем оценки качества.

## Архитектура: ГИБРИДНЫЙ ПОДХОД

Выбран гибридный подход по следующим причинам:
- Научная новизна (комбинация методов)
- Практическая надёжность
- Воспроизводимость результатов

Компоненты:
- **CV (YOLOv8)** — детекция геометрии (классы, линии, стрелки)
- **OCR (PaddleOCR)** — распознавание текста
- **Правила UML** — валидация и разрешение связей

## Пайплайн обработки

Последовательность сервисов:

ImageLoader → ElementDetector → TextParser → RelationResolver → JsonExporter

### Описание сервисов

1. **ImageLoader** — загрузка изображения, нормализация до 1024x1024
2. **ElementDetector** — YOLOv8 для детекции прямоугольников (классы) и линий (связи)
3. **TextParser** — PaddleOCR + regex для парсинга имя/атрибуты/методы
4. **RelationResolver** — геометрическое сопоставление линий с классами по BBox
5. **JsonExporter** — сериализация в JSON по схеме

## Модели данных (Pydantic v2)

Основные сущности:

- **BBox**: x_min, y_min, x_max, y_max (float)
- **Attribute**: name, type (Optional), visibility (enum: +/-/#/~)
- **Method**: name, return_type, parameters (list), visibility
- **UmlClass**: name, bbox, attributes, methods, stereotype (Optional)
- **ParsedBlock**: bbox, raw_text, parsed_name, parsed_attributes, parsed_methods
- **Relation**: source_class_id, target_class_id, relation_type (enum), multiplicities

Полная схема JSON описана в docs/json-schema.md.

## Технологический стек

Основные зависимости:
- Python 3.11
- PyTorch 2.x
- ultralytics (YOLOv8)
- paddleocr
- opencv-python
- pydantic>=2.0
- pyyaml
- numpy
- pillow

Dev-зависимости:
- pytest
- pytest-cov
- ruff
- mypy

## Ограничения версии 0.1

Входные данные:
- Путь к изображению жёстко задан в коде (константа)
- Только формат PNG
- Только UML class diagrams (не use case, не sequence)
- Только печатные диаграммы (не рукописные)

Выходные данные:
- Только JSON (схема в docs/json-schema.md)
- Нет XMI, PlantUML
- Нет визуализации результата

## Метрики успеха (целевые)

Для детекции:
- mAP@0.5 для детекции классов: > 0.85

Для распознавания:
- F1 для распознавания атрибутов/методов: > 0.80
- F1 для определения типов связей: > 0.75

Для производительности:
- Inference time: < 2 сек на изображение (GPU)
- Inference time: < 10 сек на изображение (CPU)

## Структура проекта

Ожидаемая структура после инициализации:

```
uml-recognition/
├── README.md
├── pyproject.toml
├── configs/
│   └── default.yaml
├── src/
│   └── uml_recognition/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── services/
│       │   ├── image_loader.py
│       │   ├── element_detector.py
│       │   ├── text_parser.py
│       │   ├── relation_resolver.py
│       │   └── json_exporter.py
│       └── models/
│           ├── uml_class.py
│           ├── parsed_block.py
│           ├── bbox.py
│           ├── attribute.py
│           ├── method.py
│           └── relation.py
├── tests/
├── data/
│   ├── raw/
│   └── processed/
└── docs/
    ├── context.md (этот файл)
    ├── architecture.md
    ├── requirements.md
    └── json-schema.md
```

## Правила кода

Обязательные требования:
- Google style для docstrings
- Type hints везде (включая return types)
- Pydantic v2 (НЕ v1!)
- Все модели должны иметь методы model_dump_json() и model_validate_json()
- Conventional commits (feat:, fix:, docs:)
- Нет внешних API вызовов в v0.1
- Функции не длиннее 50 строк

## Что НЕ делать

Запрещённые практики:
- Не использовать Pydantic v1
- Не смешивать бизнес-логику и IO
- Не хардкодить пути (кроме v0.1 для изображения)
- Не писать монолитные функции > 50 строк
- Не игнорировать type hints
- Не использовать внешние API в базовой версии

## Аналитическая база

Ключевые аналоги из Lab2:
- **NEURAL-UML**: 92-94% точность, открытый датасет
- **CDE**: YOLOv5 + EasyOCR + Hough transform
- **Img2UML**: классический CV-пайплайн, 4 этапа
- **MBSE-агенты**: GPT-5 F1=0.96, Qwen3-VL F1=0.94
- **Diagram Analysis (HAL)**: YOLOv8 + двойной OCR, mAP=0.91

Что берём для своего проекта:
- Архитектуру пайплайна из Img2UML/CDE
- YOLOv8 как backbone (как в Diagram Analysis)
- PaddleOCR вместо EasyOCR (лучше на техническом тексте)
- Геометрическое разрешение связей (как в CDE)

## Научные школы

Ближайшие по духу (из MNP_Lab3):
- University of Málaga (Cámara et al.) — оценка LLM в UML
- Politecnico di Torino — образовательный контекст
- University of Luxembourg/LIST — image-to-UML
- University of Twente — автоматизированная оценка

Российских устойчивых школ по этой теме нет — это наш шанс занять нишу.

## План работ
Режим разработки (Sprint до 1 декабря)
Дедлайн: 1 декабря 2026 года.
Цель: Рабочий MVP, который принимает PNG и выдает валидный JSON.
Приоритет: Скорость и работающий энд-ту-энд пайплайн.
Минимум сложных абстракций, максимум работающего кода. Если что-то блокирует — упрощаем и рефакторим потом.