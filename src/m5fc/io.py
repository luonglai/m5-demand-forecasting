"""Spark session and table read/write helpers.

The same code runs locally (Parquet folders) and on Databricks (Unity Catalog
Delta tables); `storage.mode` in the config decides which.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from pyspark.sql import DataFrame, SparkSession


def get_spark(app_name: str = "m5fc") -> SparkSession:
    """Return the active session on Databricks, or start a small local one."""
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return SparkSession.builder.getOrCreate()
    # Python workers must use the same interpreter (and installed packages) as the driver.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    return (
        SparkSession.builder.master(os.environ.get("SPARK_MASTER", "local[*]"))
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def table_ref(cfg: dict[str, Any], name: str) -> str:
    s = cfg["storage"]
    if s["mode"] == "table":
        return f"{s['catalog']}.{s['schema']}.{name}"
    return os.path.join(s["base_path"], name)


def write_table(df: DataFrame, cfg: dict[str, Any], name: str) -> None:
    ref = table_ref(cfg, name)
    writer = df.write.mode("overwrite")
    if cfg["storage"]["mode"] == "table":
        writer.option("overwriteSchema", "true").saveAsTable(ref)
    else:
        writer.parquet(ref)


def read_table(spark: SparkSession, cfg: dict[str, Any], name: str) -> DataFrame:
    ref = table_ref(cfg, name)
    if cfg["storage"]["mode"] == "table":
        return spark.read.table(ref)
    return spark.read.parquet(ref)


def read_raw_csv(spark: SparkSession, cfg: dict[str, Any], filename: str) -> DataFrame:
    path = os.path.join(cfg["storage"]["raw_path"], filename)
    return spark.read.option("header", True).option("inferSchema", True).csv(path)


def ensure_storage(spark: SparkSession, cfg: dict[str, Any]) -> None:
    """Create the Unity Catalog schema and raw-file volume if missing (no-op locally)."""
    s = cfg["storage"]
    if s["mode"] != "table":
        os.makedirs(s["base_path"], exist_ok=True)
        return
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {s['catalog']}.{s['schema']}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {s['catalog']}.{s['schema']}.raw")
