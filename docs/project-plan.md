# ShopPilot: план реализации проекта

Статус: **data platform завершена; начинается retrieval/recommendation этап**  
Горизонт: **4 последовательных delivery-этапа**  
Формат: **production-like portfolio MVP с измеримыми quality gates**

Текущий прогресс на 2026-11-06:

- Bronze и Silver опубликованы и покрыты data-quality проверками;
- Gold snapshot `amazon-all-beauty-2023-v1` опубликован по новому контракту;
- global temporal protocol, point-in-time user features и rerun signatures
  проверены на полном датасете;
- performance audit завершён: small-files defect исправлен, полный Gold
  rerun ускорен с 102.77 до 92.58 секунды в локальном before/after замере;
- runtime-код Silver/Gold и переиспользуемые Bronze helpers перенесены в
  пакет `src/shoppilot`, тесты отделены от локальных экспериментов;
- rerun signatures расширены до всех публикуемых полей и различают null,
  пустую строку и заполненное значение;
- downstream-код использует только global calendar Gold contract;
- retrieval, ALS, agent, orchestration и deployment ещё не реализованы.

Связанные документы:

- [Product & Engineering Design Document](design-document.md)
- [Recommender data contract](recommender-data-contract.md)

## 1. Цель

Создать портфолио-проект AI Engineer — B2B-платформу для интернет-магазинов косметики, которая объединяет:

- batch data platform на PySpark;
- полнотекстовый и семантический поиск;
- персональные рекомендации;
- hybrid retrieval и reranking;
- AI-консультанта с подтверждением ответов данными каталога и отзывами;
- оркестрацию в Airflow;
- управление ML/GenAI lifecycle в MLflow;
- FastAPI и демонстрационный пользовательский интерфейс;
- локальное и удалённое production-like развёртывание.

Демонстрационный датасет — категория `All_Beauty` из Amazon Reviews 2023.
Полный корпус Amazon Reviews для MVP не требуется.

## 2. Ожидаемый инженерный результат

После завершения проект должен демонстрировать следующие прикладные
компетенции:

- воспроизводимый PySpark batch pipeline с data-quality и rerun gates;
- оркестрацию batch и MLOps workflows через Airflow;
- управление ML, retrieval и GenAI lifecycle через MLflow;
- измеримые retrieval, ranking и recommendation baselines;
- framework-neutral AI-агента с grounded responses и safety gates;
- развёртывание небольшого AI-сервиса на Linux VM.

В scope не входят Spark Structured Streaming, крупные Kubernetes-кластеры,
Airflow KubernetesExecutor, enterprise IAM и эксплуатация систем с тысячами
DAG-ов.

## 3. Итоговые артефакты

К концу проекта должны существовать:

1. Bronze/Silver/Gold data pipeline на PySpark.
2. Popularity, item-to-item и implicit ALS baselines.
3. BM25, dense и hybrid retrieval.
4. CPU reranker.
5. Offline evaluation harness и проверенные evaluation datasets.
6. Собственный небольшой agent graph без обязательной зависимости от LangGraph.
7. LangGraph adapter и compatibility tests.
8. FastAPI с search, recommendations, chat, products и events API.
9. Пять Airflow DAG-ов.
10. MLflow Tracking, Registry, prompt/evaluation/tracing workflow.
11. Streamlit demo.
12. Docker Compose для локального запуска.
13. CI-проверки.
14. Инструкция и конфигурация развёртывания на Linux VM.
15. Заполненный evaluation report с техническими результатами и бизнес-гипотезами.

## 4. Границы MVP

### Входит

- Один демонстрационный tenant `beauty-demo`.
- Обязательный `tenant_id` во всех внешних контрактах и индексах.
- Каталог и отзывы `All_Beauty`.
- Локальные CPU embeddings.
- OpenAI-compatible API только для генерации ответа агента.
- Grounded ответы исключительно по товарам и evidence, возвращённым инструментами.
- Offline ML/IR evaluation.
- Сбор событий, необходимых для будущего коммерческого A/B-теста.
- Локальный запуск и развёртывание на одной удалённой VM.

### Не входит

- Полноценный multi-tenant SaaS control plane.
- Платежи и биллинг.
- Реальная интеграция с корзиной интернет-магазина.
- Обучение собственной LLM.
- GPU-инфраструктура.
- Заявления о росте конверсии без реального A/B-трафика.
- Enterprise SSO, RBAC и compliance certification.

## 5. План по неделям

### Неделя 1 — PySpark и Data Platform

- Изучить execution model Spark, DataFrame API, schemas, Catalyst и AQE.
- Загрузить и исследовать Amazon Reviews 2023 `All_Beauty`.
- Создать Bronze, Silver и Gold слои.
- Реализовать дедупликацию, оконные функции, temporal split и data-quality checks.
- Научиться читать physical plan и устранять лишние shuffle.

Результат недели: воспроизводимый и идемпотентный batch pipeline.

### Неделя 2 — Retrieval, recommendations и ranking

- Создать popularity, category popularity, item-to-item и ALS baselines.
- Построить sparse и dense индексы.
- Реализовать hybrid fusion и cross-encoder reranking.
- Подготовить минимум 150 проверенных retrieval queries.
- Сравнить подходы по quality, latency и coverage.

Результат недели: измеримый retrieval/recommendation pipeline с promotion gate.

### Неделя 3 — MLflow, Airflow и AI-агент

- Настроить MLflow Tracking, artifacts, Registry, aliases и rollback.
- Освоить MLflow GenAI traces, prompt versions и evaluation datasets.
- Создать Airflow DAG-и, assets, pools, mapping, sensors и backfill.
- Реализовать framework-neutral agent state graph и инструменты.
- Подключить FastAPI и трассировку.

Результат недели: воспроизводимый ML/GenAI lifecycle и оркестрированный end-to-end pipeline.

### Неделя 4 — Бизнес-MVP и production-like deployment

- Подготовить минимум 75 проверенных agent scenarios.
- Сравнить не менее трёх версий prompt.
- Реализовать Streamlit demo и business events.
- Добавить CI, безопасность, метрики, backups и failure drills.
- Развернуть проект на Ubuntu VM.
- Заполнить evaluation report и подготовить демонстрацию.

Результат недели: доступный по HTTPS портфолио-MVP с честно измеренными ограничениями.

## 6. Quality gates

### Data

- Явные схемы входных данных.
- Отсутствие пересечения train, validation и test.
- Повторный запуск не создаёт дубликаты.
- Partition pruning подтверждён physical plan.

### Retrieval

- Hybrid candidate улучшает лучший одиночный baseline минимум на 2% по NDCG@10.
- Recall@20 не ухудшается более чем на 1%.
- Нет повторов и cross-tenant результатов.
- Search p95 на целевой VM меньше 500 мс.

### Agent

- 100% возвращённых product IDs существуют в каталоге нужного tenant.
- Tool-selection accuracy не ниже 90%.
- Evidence citation rate не ниже 95%.
- Groundedness не ниже 85%.
- Constraint satisfaction не ниже 90%.
- Нет критических prompt-injection и tenant-isolation failures.

### Operations

- Failed quality gate не публикует модель.
- Повторный Airflow task не повреждает данные.
- Model alias можно откатить без переобучения.
- Система корректно деградирует при недоступности LLM.
- Данные и сервисы восстанавливаются из backup.

## 7. Правила работы над проектом

- Каждая итерация заканчивается работающим артефактом, тестом или измерением.
- Notebook разрешён для исследования, но production-логика переносится в Python modules/jobs.
- Любое сравнение моделей фиксируется в MLflow.
- Любая рекомендация агента должна быть проверяема через catalog data или review evidence.
- Offline-метрики и коммерческие KPI никогда не смешиваются.
- Новая сложность добавляется только после работающего baseline.
- LangGraph остаётся заменяемым runtime, а не частью domain layer.

## 8. Условия начала реализации

Перед первым днём нужно подготовить:

- 120 часов календарного времени на четыре недели;
- Git-репозиторий;
- Docker и Docker Compose;
- Python 3.12 и Java 17/21, совместимую с выбранной версией Spark;
- 80–120 GB свободного диска для образов, данных и артефактов;
- OpenAI-compatible API key с небольшим лимитом расходов;
- Linux VM к началу четвёртой недели.

Версии зависимостей фиксируются lock-файлом в первый день реализации после проверки совместимости актуальных stable-релизов.
