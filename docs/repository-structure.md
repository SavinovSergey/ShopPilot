# Repository structure and code ownership

Статус: **готово к первому Git commit**  
Дата: **2026-11-09**

## Источник истины

Канонический код проекта находится в `src/shoppilot`:

- `data/bronze` — explicit schemas и детерминированные ingestion helpers;
- `data/silver` — нормализация, дедупликация и review evidence;
- `data/gold` — facts, signals, global calendar protocol, point-in-time
  features, product documents и quality gates;
- `data/signatures.py` — null-sensitive rerun signatures;
- `jobs/silver.py` и `jobs/gold.py` — исполняемые batch jobs.

Любой новый код для recommendations, retrieval, API или orchestration должен
импортировать только `shoppilot`. Черновики и локальные эксперименты исключены
из публичного репозитория.

## Что не хранится в Git

- исходные Amazon JSONL;
- Bronze/Silver/Gold Parquet outputs и quality metrics;
- `.venv`, Spark metastore, logs и Python caches;
- локальные Codex/IDE metadata;
- secrets и `.env`.

В `data` хранится только `.gitkeep`. Инструкция получения исходных данных и
воспроизводимого полного ingestion job должна быть добавлена до публичного
релиза; текущий следующий этап может воспроизводимо читать уже опубликованный
versioned Gold snapshot.

## Проверки перед коммитом

1. `.venv/bin/python -m pytest`.
2. `PYTHONPATH=src .venv/bin/python -m shoppilot.jobs.gold` для полного rerun.
3. Убедиться через `git status --short`, что локальные данные, окружение,
   Spark metastore и учебные каталоги не предлагаются к коммиту.
