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
- Опубликован versioned model-input snapshot для validation cutoff: train-only
  пары, collision-free user/item mappings, candidate universe, seen items и
  popularity counts. Leakage и mapping gates равны нулю, два полных запуска
  дали одинаковые signatures.
- Реализован offline recommender evaluation harness: global и category popularity
  baselines, Recall/HitRate/NDCG/MRR@20, catalog coverage, novelty и отчёты по
  availability slices. Результаты сохраняются как versioned Parquet artifacts.
- На опубликованном snapshot обе модели получили Recall@20 `0.035657` при
  43 568 оцениваемых пользователях; категориальная модель не улучшила качество.
  Подробности и ограничения — в [отчёте об оценке](docs/recommender-evaluation-report.md).
- Набор из 107 тестов проверяет data pipeline, model inputs и evaluation.
- Следующий этап — локальный MLflow Tracking для модельных экспериментов.

## Структура

```text
src/shoppilot/       канонический код продукта
  data/              schemas, transforms, contracts и quality gates
  recommendations/   model inputs, baselines, metrics и evaluation
  jobs/              исполняемые Spark jobs и публикация артефактов
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
.venv/bin/python -m shoppilot.jobs.recommender_model_input
```

Последняя команда читает versioned Gold snapshot и публикует model-input
артефакты для validation cutoff `2021-01-01`. Повторный запуск сравнивает
сохранённые и заново рассчитанные signatures до перезаписи.

Оценка рекомендаций вызывается через
`shoppilot.jobs.recommender_evaluation.run_recommender_evaluation_job(spark, config, k=20)`.
Она читает опубликованные Gold и model-input artifacts и сохраняет два baseline
под `data/lake/recommender_evaluations/<snapshot_id>/<config_hash>/k=20/`.
Пути отчётов неизменяемы: повторная запись в уже существующий путь отклоняется.

Данные, виртуальное окружение, Spark metastore и сгенерированные Parquet-файлы
намеренно исключены из Git. В репозиторий входят код, тесты, документация и
пустой маркер `data/.gitkeep`.

## Основные документы

- [Product & Engineering Design](docs/design-document.md)
- [Recommender data contract](docs/recommender-data-contract.md)
- [Recommender evaluation report](docs/recommender-evaluation-report.md)
- [План реализации проекта](docs/project-plan.md)
- [Структура репозитория](docs/repository-structure.md)
