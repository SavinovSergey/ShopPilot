# Recommender data and evaluation contract

Статус: **Gold facts/signals/protocol реализованы и опубликованы**  
Версия: **0.4**  
Дата: **2026-11-06**

## 1. Зачем меняется текущая схема

Текущие `interactions_train`, `interactions_validation` и `interactions_test`
одновременно содержат исходный review, модельную интерпретацию rating и split.
Это связывает Gold с одним ALS baseline и не позволяет безопасно менять signals
или evaluation protocol.

Новая схема разделяет четыре ответственности:

```text
Silver reviews
    -> Gold user_item_events
    -> Gold interaction_signals
    -> Gold evaluation_assignments
    -> versioned model inputs and artifacts
```

Источником истины является versioned snapshot
`amazon-all-beauty-2023-v1`.

## 2. Фактический профиль данных

Локальный аудит Silver на 2026-11-06:

- всего reviews: `694252`;
- rating 1: `100888`;
- rating 2: `42601`;
- rating 3: `55720`;
- rating 4: `78608`;
- rating 5: `416435`;
- всего users: `631986`;
- users без positive rating: `176400`;
- диапазон event time: `2000-11-01` — `2023-09-09`.

Reviews являются proxy поведения, а не фактическими orders. `review_rating`
должен быть явным `event_type`; будущие click/cart/purchase/return не смешиваются
с ним без signal mapping.

## 3. Gold facts: `user_item_events`

Минимальный контракт:

```text
tenant_id
event_id
user_id
product_id
event_type
event_time
rating
verified_purchase
helpful_vote
source
source_record_id
snapshot_id
```

Правила:

- business key: `(tenant_id, event_id)`;
- сохраняются все ratings, а не только rating >= 4;
- review title/text остаются в Silver и review evidence;
- `helpful_vote` сохраняется только как review/evidence attribute и не участвует
  в recommender signals, features, splits или metrics;
- facts не содержат `preference`, `confidence`, `split` или постоянный segment;
- tenant входит во все keys, joins, windows, mappings и signatures;
- факты не удаляются из-за отсутствия product в текущем active catalog; orphan
  получает quality flag и исключается только из нужного model input.

Для production events дополнительно потребуется `ingested_at`/`available_at`,
чтобы late-arriving event не был доступен модели задним числом. В статическом
Amazon snapshot это ограничение фиксируется явно.

## 4. Model signals: `interaction_signals`

Контракт:

```text
tenant_id
event_id
user_id
product_id
event_time
signal_name
signal_version
feedback_class
preference
confidence
is_training_eligible
snapshot_id
```

Начальные версии:

### `rating_implicit_uniform_v1`

- rating >= 4: `positive`, preference `1`, confidence `1`, eligible;
- rating == 3: `neutral`, preference/confidence null, not eligible;
- rating <= 2: `negative`, preference/confidence null, not eligible.

Непросмотренная пара и явно низкая rating — разные состояния. Negative rating не
передаётся в implicit ALS как negative confidence.

## 5. User features

Features строятся для всех users и имеют `feature_version`, `snapshot_id` и
`feature_cutoff`:

```text
all_interaction_count
all_distinct_event_time_count
positive_interaction_count
positive_distinct_event_time_count
neutral_interaction_count
negative_interaction_count
first_event_time
last_event_time
first_positive_event_time
last_positive_event_time
```

`cold`, `short_history` и `warm` не хранятся как вечное свойство пользователя.
В evaluation assignment записывается `history_depth_before_cutoff`.

Availability states:

- `platform_cold`: до cutoff нет никаких events;
- `als_cold_nonpositive_only`: events есть, но positive history отсутствует;
- `short_positive_history`: один positive timestamp;
- `warm_positive_2_4`: два-четыре positive timestamps;
- `warm_positive_5_plus`: минимум пять positive timestamps.

## 6. Global temporal protocol

Основной protocol `global_calendar_v1`:

```text
train       event_time < 2021-01-01
validation  2021-01-01 <= event_time < 2022-01-01
test        2022-01-01 <= event_time < 2023-01-01
oot_stress  2023-01-01 <= event_time
```

2023 неполный и не объединяется с test.

Validation содержит `81259` positive target events и `76308` target users.
Test содержит `40160` positive target events и `37928` target users.

User states среди test target users:

- new to platform: `34939`;
- known platform user, но только non-positive history: `574`;
- один positive history timestamp: `1751`;
- минимум два positive history timestamps: `664`.

### Model selection

1. Hyperparameters, signal version, aggregation и candidate rules выбираются на
   rolling backtests и validation, без test.
2. После фиксации решения модель переобучается на всех eligible events с
   `event_time < 2022-01-01`.
3. Test-2022 запускается один раз и не влияет на следующую итерацию выбора.
4. Если test использован для принятия решения, он становится validation и для
   честного test требуется более поздний нетронутый период.

Rolling backtests используют calendar folds 2018, 2019, 2020 и 2021: модель в
каждом fold обучается только на событиях до начала соответствующего года.
Метрики агрегируются по folds; строки разных folds не объявляются независимыми.

## 7. Evaluation targets и slices

`evaluation_assignments` содержит:

```text
tenant_id
event_id
signal_version
evaluation_protocol
protocol_version
fold_id
role
cutoff_time
history_depth_before_cutoff
target_bucket_id
user_availability_state
item_availability_state
snapshot_id
```

Все events одного `(tenant_id, user_id, event_time)` имеют общий
`target_bucket_id`. Target может быть множеством товаров.

Item state для ALS определяется только по positive train history:

- `known_to_als`;
- `unknown_to_als`.

В test 2022 имеется `15110` positive target events по `6538` products, которых
нет в positive history до 2022. Это не обязательно новый catalog item; корректное
название — `unknown_to_als item`.

Отчёт публикует как минимум:

- known user + known item: conditional ALS quality;
- cold/short user + known item: popularity или personalized fallback;
- known user + unknown item: content/hybrid item cold start;
- cold user + unknown item: end-to-end fallback;
- all eligible targets: итоговое качество router + fallbacks.

Основная статистическая единица — пользователь, а не отдельный event. Несколько
targets одного пользователя образуют один relevance set для соответствующего
cutoff. Primary metrics считаются как macro average по users, а confidence
intervals — user-level bootstrap. Event-level micro metrics разрешены только как
дополнительные и явно подписываются.

Начальный `global_calendar_v1` оценивает frozen snapshot на следующем calendar
window. Если production cadence станет weekly/monthly, добавляется отдельный
rolling protocol: logic и hyperparameters остаются зафиксированными, а каждый
fold обучается только на данных до начала своего окна. Его результаты нельзя
смешивать с frozen-snapshot protocol в одну метрику.

## 8. Model-ready ALS input

ALS input не равен event table. После применения cutoff и signal filter:

1. повторные events агрегируются до одной строки `(tenant_id, user_id, product_id)`;
2. правило aggregation и cap confidence версионируются;
3. user/product string IDs отображаются в стабильные integer indices;
4. mapping tables являются artifacts конкретного model snapshot;
5. popularity, category popularity и seen items считаются только из train facts;
6. validation/test targets не входят в pair aggregation.

Хеширование string ID прямо в integer без collision control запрещено.

## 9. Leakage и risk controls

На версии 0.4 реализованы Gold facts, uniform signals, point-in-time user
features и `global_calendar_v1`. Полный job выполнен дважды; signatures всех
пяти datasets совпали. Downstream-код использует только versioned snapshot,
signal и protocol contracts.

На следующем этапе остаются ALS-specific gates: pair aggregation, stable integer
mappings, train-only popularity/candidate universe и честный расчёт model metrics.

Перед публикацией model input проверяется:

- максимальный `available_at` train facts/features меньше cutoff;
- никаких validation/test IDs в train;
- user/item counts, popularity и mappings построены только по train;
- segmentation использует только историю до cutoff;
- same-timestamp bucket не разделён;
- signal/config/protocol/snapshot versions записаны в output;
- unknown users/items не подаются в ALS как известные;
- test не используется для hyperparameter/model selection;
- negative/neutral events не превращены в unobserved zeros;
- повторные user-item events агрегированы детерминированно;
- candidate generation применяет tenant filter;
- primary top-K metrics считаются по полному eligible candidate universe;
  sampled negatives допустимы только как отдельно подписанный diagnostic;
- candidate universe, cutoffs и protocol config hash фиксируются до первого
  просмотра test model metrics;
- confidence intervals строятся по users, а не по зависимым events;
- dataset signatures стабильны для повторного запуска одного snapshot.

Catalog availability point-in-time восстановить полностью нельзя: metadata не
содержит надёжную историю появления/исчезновения SKU. ALS evaluation ограничивается
items, известными train. Content/new-item evaluation публикуется с этой оговоркой.

Статусы рисков после реализации разделяются так:

- устраняемые quality gates: cross-user future leakage, target overlap,
  future-based segmentation, test tuning, same-timestamp split, tenant leakage;
- контролируемые и измеряемые: sparse warm sample, unknown users/items,
  popularity bias, repeat-purchase policy, model staleness;
- неустранимые на Amazon Reviews snapshot: review вместо purchase, отсутствие
  impression/exposure logs, неизвестная historical catalog availability,
  неизвестное позднее поступление events.

### 9.1 Фактический residual-risk audit

Аудит на train `< 2022-01-01` и test-2022:

- все `112578` product records имеют `updated_at = 2026-10-22`, поэтому это
  ingestion timestamp, а не historical availability;
- `94875` products (`84.3%`) не имеют price;
- `82` review events относятся к `12` products, отсутствующим в catalog;
- в test это `11` positive events по `2` orphan products;
- train содержит `445101` positive events и `444894` distinct user-item pairs;
- повторяются только `193` train pairs (`400` events, максимум `5` на пару);
- test содержит только `16` repeat-positive targets;
- multi-event user timestamp buckets в test отсутствуют, но invariant остаётся;
- train содержит `83533` known-to-ALS items;
- top 1% train items получает `26.6%` positive events, top 10% — `61.1%`;
- в test `15110` из `40160` positive targets (`37.6%`) относятся к
  unknown-to-ALS items;
- из `664` warm test users только `307` имеют хотя бы один known-to-ALS target;
- `36645` из `37928` test users (`96.6%`) имеют ровно один target item;
- только `611` test target users имеют хотя бы один prior negative item.

Следствия:

- основной ALS report обязан показывать sample size и conditional known-item
  metrics; общей test-метрики недостаточно;
- end-to-end report обязан включать fallback, иначе 37.6% targets недостижимы;
- popularity оценивается вместе с coverage/novelty/long-tail metrics;
- repeat-purchase и negative-avoidance результаты считаются exploratory из-за
  малого sample;
- orphan targets не удаляются молча: публикуется отдельная eligibility metric;
- price-aware сценарии оцениваются только на явно указанном price-covered slice.

Historical product availability восстановить нельзя. Для interaction-only
baselines используется conservative catalog: items, наблюдавшиеся до cutoff.
Content/new-item evaluation на current metadata маркируется как retrospective и
не объявляется строгой point-in-time оценкой.

## 10. Seen-item и repeat-purchase policy

Для beauty повторная покупка может быть полезной рекомендацией. Поэтому:

- targets делятся на first-time и repeat interactions;
- метрики считаются отдельно;
- seen-item exclusion является versioned serving/evaluation parameter;
- target, исключённый candidate policy, не должен незаметно попадать в denominator:
  причина exclusion публикуется отдельной quality metric.

## 11. Snapshot и публикация

Идемпотентность относится к одному `snapshot_id`. Новые Silver facts создают новый
Gold snapshot и не считаются rerun mismatch.

Внутри snapshot datasets partitioned по `tenant_id`. На текущем объёме не нужно
добавлять partitions по user segment или split: это создаст small files.

Рекомендуемая организация:

```text
gold/snapshots/<snapshot_id>/user_item_events
gold/snapshots/<snapshot_id>/user_features
gold/snapshots/<snapshot_id>/product_documents
gold/snapshots/<snapshot_id>/interaction_signals/<signal_version>
gold/snapshots/<snapshot_id>/evaluation_assignments/<protocol_version>
```

MLflow run позднее хранит snapshot ID, signatures, signal/protocol versions,
cutoffs, aggregation version и integer mapping artifacts.

## 12. Статус миграции и следующий этап

Выполнено:

1. Gold transforms и contract tests перенесены в канонический пакет.
2. `user_item_events` опубликован с parity `694252 / 694252`.
3. `rating_implicit_uniform_v1` опубликован без зависимости от `helpful_vote`.
4. User features опубликованы для cutoffs 2021, 2022 и 2023.
5. `global_calendar_v1` и dynamic availability states реализованы.
6. Temporal/leakage gates прошли; полный rerun signatures совпал.
7. Неканонические model inputs исключены из downstream dependencies.

Следующий этап:

1. Агрегировать train positive events до user-item pairs по versioned policy.
2. Создать collision-free stable integer mappings как model artifacts.
3. Построить popularity, category-popularity и item-to-item baselines только из
   истории до cutoff.
4. Обучить ALS и считать conditional known-item и end-to-end fallback metrics
   раздельно.
5. Сравнить ALS с popularity и item-to-item baselines в одном versioned
   evaluation protocol.
