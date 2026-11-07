# Схема выходного JSON и Pydantic v2 модели

Этот файл описывает точную структуру выходных данных модуля. 
Все модели данных должны быть реализованы на Python с использованием Pydantic v2 (BaseModel).

## 1. Полный пример выходного JSON

```json
{
  "classes": [
    {
      "name": "User",
      "bbox": {
        "x_min": 100.0,
        "y_min": 50.0,
        "x_max": 300.0,
        "y_max": 250.0
      },
      "stereotype": null,
      "attributes": [
        {
          "name": "id",
          "type": "int",
          "visibility": "-"
        }
      ],
      "methods": [
        {
          "name": "login",
          "return_type": "bool",
          "parameters": [
            {
              "name": "password",
              "type": "str"
            }
          ],
          "visibility": "+"
        }
      ]
    }
  ],
  "relations": [
    {
      "source_class_id": "User",
      "target_class_id": "Order",
      "relation_type": "association",
      "multiplicity_source": "1",
      "multiplicity_target": "*"
    }
  ],
  "metadata": {
    "source_image": "data/raw/example.png",
    "processing_time_ms": 1523.45,
    "schema_version": "1.0"
  }
}
```

## 2. Референсные Pydantic v2 модели

Используй этот код как основу для файлов в `src/uml_recognition/models/`.

```python
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

# --- Enums ---

class Visibility(str, Enum):
    PUBLIC = "+"
    PRIVATE = "-"
    PROTECTED = "#"
    PACKAGE = "~"

class RelationType(str, Enum):
    ASSOCIATION = "association"
    INHERITANCE = "inheritance"
    DEPENDENCY = "dependency"
    AGGREGATION = "aggregation"
    COMPOSITION = "composition"

# --- Базовые сущности ---

class BBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

class Parameter(BaseModel):
    name: str
    type: Optional[str] = None

class Attribute(BaseModel):
    name: str
    type: Optional[str] = None
    visibility: Visibility = Visibility.PUBLIC

class Method(BaseModel):
    name: str
    return_type: Optional[str] = None
    parameters: List[Parameter] = Field(default_factory=list)
    visibility: Visibility = Visibility.PUBLIC

class UmlClass(BaseModel):
    name: str
    bbox: BBox
    stereotype: Optional[str] = None
    attributes: List[Attribute] = Field(default_factory=list)
    methods: List[Method] = Field(default_factory=list)

class Relation(BaseModel):
    source_class_id: str
    target_class_id: str
    relation_type: RelationType
    multiplicity_source: Optional[str] = None
    multiplicity_target: Optional[str] = None

# --- Метаданные и корневая модель ---

class Metadata(BaseModel):
    source_image: str
    processing_time_ms: float
    schema_version: str = "1.0"

class UmlDiagram(BaseModel):
    classes: List[UmlClass] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    metadata: Metadata
```

## 3. Правила валидации

1. Все координаты BBox должны быть >= 0.
2. x_min должно быть строго меньше x_max.
3. y_min должно быть строго меньше y_max.
4. Поле `name` в UmlClass, Attribute, Method не может быть пустой строкой.
5. Поле `source_class_id` и `target_class_id` в Relation должны ссылаться на существующие имена классов из списка `classes` (валидация выполняется на уровне Pipeline, а не внутри модели Relation, так как Relation создается до финальной сборки).