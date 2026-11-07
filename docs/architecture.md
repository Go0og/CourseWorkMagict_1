# Архитектура системы

## Компонентная диаграмма

```text
+---------------------------------------------------------+
|                    Pipeline (Оркестратор)                |
|  - Управляет потоком данных                             |
|  - Вызывает сервисы последовательно                     |
|  - Обрабатывает ошибки                                  |
+---------------------------------------------------------+
                            |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
+--------------+   +--------------+   +--------------+
|ImageLoader   |   |ElementDetect |   | TextParser   |
|              |   |              |   |              |
|- Загрузка    |   |- YOLOv8      |   |- PaddleOCR   |
|- Resize      |   |- BBox        |   |- Regex       |
|- Normalize   |   |- Classes     |   |- Parse       |
+--------------+   +--------------+   +--------------+
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
                  +------------------+
                  |RelationResolver  |
                  |                  |
                  |- Geometry match  |
                  |- Line classif.   |
                  +------------------+
                            |
                            v
                  +------------------+
                  |  JsonExporter    |
                  |                  |
                  |- Serialize       |
                  |- Validate schema |
                  +------------------+
```

## Поток данных

1. **ImageLoader** принимает путь к PNG -> возвращает `np.ndarray` + мета-данные (оригинальные размеры).
2. **ElementDetector** принимает изображение -> возвращает список `BBox` для классов и линии для связей.
3. **TextParser** для каждого `BBox` класса -> возвращает `ParsedBlock` с именем, атрибутами, методами.
4. **RelationResolver** сопоставляет линии с `BBox` классов -> возвращает список `Relation`.
5. **JsonExporter** собирает всё в итоговый JSON (модель `UmlDiagram`).

## Модели данных (Pydantic v2)

### BBox
- x_min: float
- y_min: float
- x_max: float
- y_max: float
- Свойства: width, height, area

### Attribute
- name: str
- type: Optional[str]
- visibility: Enum ("+", "-", "#", "~")

### Parameter
- name: str
- type: Optional[str]

### Method
- name: str
- return_type: Optional[str]
- parameters: List[Parameter]
- visibility: Enum ("+", "-", "#", "~")

### UmlClass
- name: str
- bbox: BBox
- stereotype: Optional[str]
- attributes: List[Attribute]
- methods: List[Method]

### Relation
- source_class_id: str
- target_class_id: str
- relation_type: Enum ("association", "inheritance", "dependency", "aggregation", "composition")
- multiplicity_source: Optional[str]
- multiplicity_target: Optional[str]

### UmlDiagram (Корневая модель)
- classes: List[UmlClass]
- relations: List[Relation]
- metadata: dict (source_image, processing_time_ms, schema_version)

## Конфигурация (configs/default.yaml)

```yaml
image:
  target_size: [1024, 1024]
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

detector:
  model: yolov8n.pt
  confidence: 0.5
  classes:
    - class_box
    - attribute_section
    - method_section
    - association_line
    - inheritance_line
    - dependency_line
    - aggregation_arrow

ocr:
  engine: paddleocr
  lang: en
  use_gpu: true

output:
  format: json
  schema_version: "1.0"
  indent: 2
```
