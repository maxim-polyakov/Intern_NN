# Inventory AI MVP

Демонстрационный модуль для нормализации складских движений, детерминированного
прогноза потребности и разбора русскоязычных вопросов. Расчёты не используют LLM
и не требуют API-ключей.

## Установка и запуск

Требуется Python 3.11+. Полный набор проверок подтверждён на Python 3.11.9.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Запуск полного пайплайна и обновление отчёта:

```bash
inventory-ai --results RESULTS.md
# Резервный вариант, если каталог Scripts ещё не попал в PATH:
python -m inventory_ai.cli --results RESULTS.md
```

Машиночитаемый вывод:

```bash
inventory-ai --json
```

Проверки:

```bash
pytest
pytest --cov=inventory_ai --cov=solution --cov-report=term-missing
ruff check .
ruff format --check .
```

Проверка сборки устанавливаемого wheel:

```bash
python -m build --wheel
```

Публичные функции доступны из `solution.py`:

- `normalize_movement(text: str) -> dict`
- `forecast_demand(history: dict, sku: str, horizon_days: int, params: dict) -> dict`
- `answer_question(question: str, context: dict) -> tuple[dict, float, str]`

## Архитектура

- `src/inventory_ai/normalization.py` — даты, SKU/названия, операции, единицы,
  составные упаковки, партии и документы;
- `src/inventory_ai/forecasting.py` — восстановление пропусков, смена уровня,
  EWMA, страховой запас и округление заказа;
- `src/inventory_ai/qa.py` — rule-based интенты, сущности, уточнения и безопасные
  ответы без выдуманных данных;
- `src/inventory_ai/reporting.py` — единый воспроизводимый pipeline;
- `dataset.json`, `catalog.json` — данные приложений к заданию;
- `tests/` — контрактные и функциональные pytest-тесты.

Решения, результаты M1–M8, шесть прогнозов и все 15 вопросов приведены в
`RESULTS.md`.

## Границы MVP

В исходных данных нет дат поставок, партий со сроками годности, истории цен и
остатков по филиалам. Система явно сообщает об отсутствии данных. Автоматическое
размещение заказа не реализовано.

## Мотивация участия в проекте

**Почему мне интересен этот проект?**

Проект интересен мне как возможность получить практические навыки в Data Science
и применить их при решении реальной задачи.

**Как я вижу свою роль в команде, которая создаёт продукт?**

Вижу себя в роли Data Scientist (DS-специалиста).

**Сколько времени в неделю я готов(а) уделять проекту и в течение какого периода?**

Готов уделять проекту 40 часов в неделю до его сдачи.
