"""Null-sensitive logical dataset signatures used by rerun gates."""

from pyspark.sql import functions as F
from pyspark.sql.types import MapType


def calculate_dataset_signature(
    dataframe,
    business_key_columns,
    checksum_columns,
) -> dict:
    missing_columns = sorted(
        set(business_key_columns + checksum_columns) - set(dataframe.columns)
    )
    if missing_columns:
        raise ValueError(
            "Cannot calculate dataset signature; missing columns: "
            + ", ".join(missing_columns)
        )

    field_types = {
        field.name: field.dataType for field in dataframe.schema.fields
    }
    canonical_columns = []
    for name in checksum_columns:
        value = F.col(name)
        if isinstance(field_types[name], MapType):
            value = F.sort_array(F.map_entries(value))
        canonical_columns.append(value.alias(name))

    serialized_value = F.to_json(
        F.struct(*canonical_columns),
        {"ignoreNullFields": "false"},
    )
    row = dataframe.agg(
        F.count("*").alias("row_count"),
        F.countDistinct(
            F.struct(*[F.col(name) for name in business_key_columns])
        ).alias("distinct_business_keys"),
        F.coalesce(
            F.sum(F.xxhash64(serialized_value).cast("decimal(38, 0)")),
            F.lit(0).cast("decimal(38, 0)"),
        ).alias("stable_checksum"),
    ).first()
    return {
        "row_count": row["row_count"],
        "distinct_business_keys": row["distinct_business_keys"],
        "duplicate_business_keys": (
            row["row_count"] - row["distinct_business_keys"]
        ),
        "stable_checksum": str(row["stable_checksum"]),
    }


def metrics_to_rows(entity: str, run_date: str, metrics: dict) -> list[tuple]:
    return [
        (entity, run_date, name, str(value) if value is not None else None)
        for name, value in metrics.items()
    ]
