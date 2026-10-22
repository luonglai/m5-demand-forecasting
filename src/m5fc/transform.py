"""Medallion layers.

bronze  raw CSVs stored as tables, unchanged
silver  one row per item x store x day: sales joined with calendar and prices,
        filtered to the configured scope, pre-launch days removed
gold    silver plus leak-free features for the ML model
"""

from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from m5fc.io import read_raw_csv, read_table, write_table

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
FEATURE_LAGS = [28, 35, 42, 49, 56]  # all >= horizon, so no recursive forecasting is needed


# --------------------------------------------------------------------- bronze
def build_bronze(spark: SparkSession, cfg: dict[str, Any]) -> None:
    for name, file in [
        ("bronze_calendar", "calendar.csv"),
        ("bronze_prices", "sell_prices.csv"),
        ("bronze_sales", "sales_train_evaluation.csv"),
    ]:
        write_table(read_raw_csv(spark, cfg, file), cfg, name)


# --------------------------------------------------------------------- silver
def filter_scope(sales: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    scope = cfg["scope"]
    if scope.get("categories"):
        sales = sales.filter(F.col("cat_id").isin(scope["categories"]))
    if scope.get("stores"):
        sales = sales.filter(F.col("store_id").isin(scope["stores"]))
    if scope.get("max_items"):
        rank = F.dense_rank().over(Window.partitionBy("dept_id").orderBy("item_id"))
        sales = sales.withColumn("_r", rank).filter(F.col("_r") <= scope["max_items"]).drop("_r")
    return sales


def melt_sales(sales_wide: DataFrame) -> DataFrame:
    """Wide d_1..d_N columns -> long rows with an integer day number."""
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    long = sales_wide.unpivot(ID_COLS, day_cols, "d", "sales")
    return long.withColumn("d_num", F.substring("d", 3, 10).cast("int")).withColumn(
        "sales", F.col("sales").cast("double")
    )


def prepare_calendar(cal: DataFrame) -> DataFrame:
    return cal.select(
        F.substring("d", 3, 10).cast("int").alias("d_num"),
        F.to_date("date").alias("date"),
        "wm_yr_wk",
        "wday",
        "month",
        "year",
        F.col("event_name_1").isNotNull().cast("int").alias("is_event"),
        F.coalesce("event_type_1", F.lit("none")).alias("event_type"),
        "snap_CA",
        "snap_TX",
        "snap_WI",
    )


def build_silver_frame(sales_wide: DataFrame, cal: DataFrame, prices: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    long = melt_sales(filter_scope(sales_wide, cfg)).drop("d")
    cal = prepare_calendar(cal)
    df = long.join(cal, "d_num", "inner")
    df = df.withColumn(
        "snap",
        F.when(F.col("state_id") == "CA", F.col("snap_CA"))
        .when(F.col("state_id") == "TX", F.col("snap_TX"))
        .otherwise(F.col("snap_WI")),
    ).drop("snap_CA", "snap_TX", "snap_WI")
    df = df.join(
        prices.select("store_id", "item_id", "wm_yr_wk", F.col("sell_price").cast("double")),
        ["store_id", "item_id", "wm_yr_wk"],
        "left",
    )
    # An item has no price before it launches in a store; those zeros are not demand.
    return df.filter(F.col("sell_price").isNotNull())


def build_silver(spark: SparkSession, cfg: dict[str, Any]) -> None:
    silver = build_silver_frame(
        read_table(spark, cfg, "bronze_sales"),
        read_table(spark, cfg, "bronze_calendar"),
        read_table(spark, cfg, "bronze_prices"),
        cfg,
    )
    write_table(silver, cfg, "silver_sales")


# ----------------------------------------------------------------------- gold
def add_features(silver: DataFrame) -> DataFrame:
    # Range windows over the day number (not row offsets), so a missing day can
    # never shift a lag. Every window ends at least 28 days back = the horizon,
    # which keeps all features known at forecast time (no leakage).
    rng = Window.partitionBy("id").orderBy("d_num")
    df = silver
    for k in FEATURE_LAGS:
        df = df.withColumn(f"lag_{k}", F.avg("sales").over(rng.rangeBetween(-k, -k)))
    df = (
        df.withColumn("rmean_7_lag28", F.avg("sales").over(rng.rangeBetween(-34, -28)))
        .withColumn("rmean_28_lag28", F.avg("sales").over(rng.rangeBetween(-55, -28)))
        .withColumn("rstd_28_lag28", F.stddev("sales").over(rng.rangeBetween(-55, -28)))
        .withColumn("rmean_91_lag28", F.avg("sales").over(rng.rangeBetween(-118, -28)))
    )
    item_store = Window.partitionBy("id")
    df = (
        df.withColumn("price_rel", F.col("sell_price") / F.avg("sell_price").over(item_store))
        .withColumn("price_change_7", F.col("sell_price") / F.avg("sell_price").over(rng.rangeBetween(-7, -7)) - 1)
        .withColumn("days_since_launch", F.col("d_num") - F.min("d_num").over(item_store))
    )
    return df


def build_gold(spark: SparkSession, cfg: dict[str, Any]) -> None:
    write_table(add_features(read_table(spark, cfg, "silver_sales")), cfg, "gold_features")
