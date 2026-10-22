"""Forecast performance: accuracy, bias and forecast value add (FVA).

  WMAPE  = sum|forecast - actual| / sum(actual)          lower is better
  bias   = sum(forecast - actual) / sum(actual)          >0 over-forecast, <0 under
  FVA    = WMAPE(naive) - WMAPE(model), in points         >0 means the model adds value

Reported at daily and weekly item x store level (planners usually judge weekly
buckets), for the total and split by store, department and demand class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from m5fc.io import read_table, write_table

LEVELS = ["total", "store_id", "dept_id", "demand_class"]


def wmape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.sum(actual)
    return float(np.sum(np.abs(forecast - actual)) / denom) if denom else float("nan")


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.sum(actual)
    return float(np.sum(forecast - actual) / denom) if denom else float("nan")


def all_forecasts(spark: SparkSession, cfg: dict[str, Any]) -> DataFrame:
    return read_table(spark, cfg, "forecast_baseline").unionByName(read_table(spark, cfg, "forecast_lightgbm"))


def with_actuals(fc: DataFrame, silver: DataFrame, classes: DataFrame) -> DataFrame:
    actual = silver.select("id", "d_num", "store_id", "dept_id", "sell_price", F.col("sales").alias("actual"))
    return fc.join(actual, ["id", "d_num"], "inner").join(
        classes.select("id", "cutoff", "demand_class"), ["id", "cutoff"], "left"
    )


def _metrics(df: DataFrame, grain: str) -> DataFrame:
    out = None
    for level in LEVELS:
        key = F.lit("all") if level == "total" else F.col(level)
        agg = (
            df.withColumn("level", F.lit(level))
            .withColumn("level_value", key)
            .groupBy("cutoff", "model", "level", "level_value")
            .agg(
                F.sum(F.abs(F.col("forecast") - F.col("actual"))).alias("abs_err"),
                F.sum(F.col("forecast") - F.col("actual")).alias("err"),
                F.sum("actual").alias("actual"),
                F.sum("forecast").alias("forecast"),
            )
            .withColumn("wmape", F.col("abs_err") / F.col("actual"))
            .withColumn("bias", F.col("err") / F.col("actual"))
            .withColumn("grain", F.lit(grain))
        )
        out = agg if out is None else out.unionByName(agg)
    return out


def compute_metrics(joined: DataFrame) -> DataFrame:
    daily = _metrics(joined, "daily")
    week = F.floor((F.col("d_num") - F.col("cutoff") - 1) / 7)
    weekly_df = joined.groupBy("cutoff", "model", "id", "store_id", "dept_id", "demand_class", week.alias("wk")).agg(
        F.sum("forecast").alias("forecast"), F.sum("actual").alias("actual")
    )
    metrics = daily.unionByName(_metrics(weekly_df, "weekly"))
    naive = metrics.filter(F.col("model") == "naive").select(
        "cutoff", "level", "level_value", "grain", F.col("wmape").alias("wmape_naive")
    )
    return metrics.join(naive, ["cutoff", "level", "level_value", "grain"], "left").withColumn(
        "fva_pts", (F.col("wmape_naive") - F.col("wmape")) * 100
    )


def run_evaluate(spark: SparkSession, cfg: dict[str, Any]) -> None:
    joined = with_actuals(
        all_forecasts(spark, cfg), read_table(spark, cfg, "silver_sales"), read_table(spark, cfg, "demand_class")
    )
    write_table(joined, cfg, "forecast_vs_actual")
    write_table(compute_metrics(joined), cfg, "forecast_metrics")
