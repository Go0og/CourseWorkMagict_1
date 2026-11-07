# UML Class Diagram Recognition Module
## Описание
Модуль для автоматического извлечения структуры UML-диаграмм классов из изображений. Принимает на вход PNG-файл (экспорт или скриншот диаграммы классов), возвращает структурированный JSON с классами, атрибутами, методами и связями.
## Контекст
Это часть магистерской диссертации : "Разработка метода оценки качества сгенерированных с помощью LLM UML-диаграмм". Модуль извлечения — второй из трёх в общей системе оценки качества.
Быстрый старт (v0.1)
python
from uml_recognition import Pipeline

pipeline = Pipeline()
result = pipeline.run()
##Выходной формат
### json
{
  "classes": [
    {
      "name": "User",
      "bbox": {"x_min": 100, "y_min": 50, "x_max": 300, "y_max": 200},
      "attributes": [{"name": "id", "type": "int", "visibility": "+"}],
      "methods": [{"name": "login", "return_type": "bool", "parameters": [], "visibility": "+"}]
    }
  ],
  "relations": [
    {
      "source": "User",
      "target": "Order",
      "type": "association",
      "multiplicity_source": "1",
      "multiplicity_target": "*"
    }
  ]
}

## Технологический стек
Python 3.11+
PyTorch 2.x
Ultralytics YOLOv8
PaddleOCR
OpenCV 4.x
Pydantic v2
## Статус разработки
Проектирование архитектуры
Инициализация проекта
Базовые модели данных
ImageLoader
ElementDetector
TextParser
RelationResolver
JsonExporter
Тестирование