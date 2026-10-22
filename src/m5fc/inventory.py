"""What better forecasts are worth: a simple safety-stock simulation.

1. On the validation fold, measure each series' forecast error (RMSE) per model.
2. Size safety stock for the target service level:  SS = z * RMSE * sqrt(lead time).
3. On the held-out test fold, hold forecast + SS each day and replay real demand.

Per model this reports the fill rate reached, units of lost sales, excess
units (a proxy for write-off risk) and the value of safety stock held
(working capital). A more accurate, less biased forecast should hit the same
service level with less stock.

This is a deliberately simple daily order-up-to approximation, meant to compare
models on equal terms, not to replace a planning system's inventory logic.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from m5fc.io import read_table, write_table


def safety_stock_factor(service_level: float, lead_time_days: int) -> float:
    return NormalDist().inv_cdf(service_level) * lead_time_days**0.5


def simulate(fva: DataFrame, cfg: dict[str, Any]) -> DataFrame:
    val_cut, test_cut = cfg["forecast"]["cutoffs"][0], cfg["forecast"]["cutoffs"][-1]
    k = safety_stock_factor(cfg["inventory"]["service_level"], cfg["inventory"]["lead_time_days"])

    sigma = (
        fva.filter(F.col("cutoff") == val_cut)
        .groupBy("id", "model")
        .agg(F.sqrt(F.avg((F.col("forecast") - F.col("actual")) ** 2)).alias("rmse_val"))
    )
    test = fva.filter(F.col("cutoff") == test_cut).join(sigma, ["id", "model"], "left")
    test = (
        test.withColumn("safety_stock", F.coalesce(F.col("rmse_val"), F.lit(0.0)) * F.lit(k))
        .withColumn("stock", F.col("forecast") + F.col("safety_stock"))
        .withColumn("served", F.least("actual", "stock"))
        .withColumn("lost", F.greatest(F.col("actual") - F.col("stock"), F.lit(0.0)))
        .withColumn("excess", F.greatest(F.col("stock") - F.col("actual"), F.lit(0.0)))
        .withColumn("in_stock", (F.col("actual") <= F.col("stock")).cast("int"))
    )
    n_days = test.select("d_num").distinct().count()
    out = None
    for level in ["total", "demand_class"]:
        key = F.lit("all") if level == "total" else F.coalesce(F.col(level), F.lit("unknown"))
        agg = (
            test.withColumn("level", F.lit(level))
            .withColumn("level_value", key)
            .groupBy("model", "level", "level_value")
            .agg(
                (F.sum("served") / F.sum("actual")).alias("fill_rate"),
                F.avg("in_stock").alias("cycle_service_level"),
                F.sum("lost").alias("lost_units"),
                F.sum("excess").alias("excess_units"),
                (F.sum("safety_stock") / F.lit(n_days)).alias("avg_safety_stock_units"),
                (F.sum(F.col("safety_stock") * F.col("sell_price")) / F.lit(n_days)).alias("avg_safety_stock_value"),
            )
        )
        out = agg if out is None else out.unionByName(agg)
    return out


def run_inventory(spark: SparkSession, cfg: dict[str, Any]) -> None:
    write_table(simulate(read_table(spark, cfg, "forecast_vs_actual"), cfg), cfg, "inventory_simulation")
