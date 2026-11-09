# ShopPilot

[![CI](https://github.com/SavinovSergey/ShopPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/SavinovSergey/ShopPilot/actions/workflows/ci.yml)

ShopPilot — production-like portfolio project AI Engineer для e-commerce: batch
data platform, рекомендации, hybrid retrieval, reranking и grounded
AI-консультант. Первый датасет — Amazon Reviews 2023 `All_Beauty`.

## Текущее состояние

- Bronze, Silver и Gold построены локально на PySpark.
- Канонический Gold snapshot использует `rating_implicit_uniform_v1` и
  `global_calendar_v1`.
- Реализованы point-in-time user features, cold/warm evaluation slices,
  quality gates и null-sensitive rerun signatures.
- Следующий этап — popularity, item-to-item и implicit ALS baselines с честной
  offline evaluation.

## Структура

```text
src/shoppilot/       канонический код продукта
  data/              schemas, transforms, contracts и quality gates
  jobs/              исполняемые Spark jobs
tests/               тесты актуального pipeline
docs/                product design, data contract и architecture notes
data/                локальные source/lake artifacts, не хранятся в Git
```

## Локальный запуск

Проект рассчитан на Python 3.12 и PySpark 4.2.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```

После подготовки локальных Bronze/Silver данных:

```bash
.venv/bin/python -m shoppilot.jobs.silver
.venv/bin/python -m shoppilot.jobs.gold
.venv/bin/python -m shoppilot.tools.performance_audit
```

Данные, виртуальное окружение, Spark metastore и сгенерированные Parquet-файлы
намеренно исключены из Git. В репозиторий входят код, тесты, документация и
пустой маркер `data/.gitkeep`.

## Основные документы

- [Product & Engineering Design](docs/design-document.md)
- [Recommender data contract](docs/recommender-data-contract.md)
- [План реализации проекта](docs/project-plan.md)
- [Структура репозитория](docs/repository-structure.md)
