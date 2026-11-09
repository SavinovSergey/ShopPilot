"""Reusable deterministic helpers for Bronze ingestion."""

from pyspark.sql import functions as F
from pyspark.sql.types import MapType


def source_record_id_expr(dataframe, source_kind: str):
    """Hash source values canonically, preserving nulls and map key order."""
    canonical_fields = []
    for field in dataframe.schema.fields:
        if field.name == "_corrupt_record":
            continue
        value = F.col(field.name)
        if isinstance(field.dataType, MapType):
            value = F.sort_array(F.map_entries(value))
        canonical_fields.append(value.alias(field.name))

    canonical_json = F.to_json(
        F.struct(*canonical_fields),
        {"ignoreNullFields": "false"},
    )
    namespace = F.lit(f"amazon-reviews-2023|{source_kind}|v1|")
    return F.sha2(F.concat(namespace, canonical_json), 256)


def add_ingestion_metadata(
    dataframe,
    *,
    tenant_id: str,
    batch_id: str,
    ingestion_date: str,
    source_kind: str,
):
    """Attach the common Bronze audit fields to valid source rows."""
    return (
        dataframe.withColumn(
            "source_record_id",
            source_record_id_expr(dataframe, source_kind),
        )
        .withColumn("tenant_id", F.lit(tenant_id))
        .withColumn("batch_id", F.lit(batch_id))
        .withColumn("source_file", F.input_file_name())
        .withColumn("ingested_at", F.current_timestamp())
        .withColumn("ingestion_date", F.lit(ingestion_date).cast("date"))
        .withColumn("schema_version", F.lit(1))
    )
