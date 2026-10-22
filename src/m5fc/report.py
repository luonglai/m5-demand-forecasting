"""Print a short results summary (and save it as Markdown on local runs)."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from m5fc.io import read_table

MODEL_ORDER = ["naive", "ets", "croston_sba", "stat_baseline", "lightgbm"]


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _order(df: pd.DataFrame) -> pd.DataFrame:
    df["_o"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return df.sort_values("_o").drop(columns="_o")


def build_report(spark: SparkSession, cfg: dict[str, Any]) -> str:
    test_cut = cfg["forecast"]["cutoffs"][-1]
    m = (
        read_table(spark, cfg, "forecast_metrics")
        .filter((F.col("cutoff") == test_cut) & (F.col("level") == "total"))
        .toPandas()
    )
    acc = m.pivot_table(index="model", columns="grain", values=["wmape", "bias", "fva_pts"]).reset_index()
    acc.columns = ["model"] + [f"{a}_{g}" for a, g in acc.columns[1:]]
    acc = _order(acc)[["model", "wmape_daily", "wmape_weekly", "bias_weekly", "fva_pts_weekly"]]
    for c in ["wmape_daily", "wmape_weekly", "bias_weekly"]:
        acc[c] = (acc[c] * 100).map("{:.1f}%".format)
    acc["fva_pts_weekly"] = acc["fva_pts_weekly"].map("{:+.1f}".format)

    cls = (
        read_table(spark, cfg, "forecast_metrics")
        .filter((F.col("cutoff") == test_cut) & (F.col("level") == "demand_class") & (F.col("grain") == "weekly"))
        .toPandas()
    )
    cls = cls.pivot_table(index="level_value", columns="model", values="wmape").reset_index()
    cls = cls.rename(columns={"level_value": "demand_class"})
    for c in cls.columns[1:]:
        cls[c] = (cls[c] * 100).map("{:.1f}%".format)

    inv = read_table(spark, cfg, "inventory_simulation").filter(F.col("level") == "total").toPandas()
    inv = _order(inv)[["model", "fill_rate", "lost_units", "excess_units", "avg_safety_stock_value"]]
    inv["fill_rate"] = (inv["fill_rate"] * 100).map("{:.1f}%".format)
    for c in ["lost_units", "excess_units", "avg_safety_stock_value"]:
        inv[c] = inv[c].map("{:,.0f}".format)

    return "\n\n".join(
        [
            f"## Test fold (days {test_cut + 1}-{test_cut + cfg['forecast']['horizon']})",
            "### Accuracy and bias, item x store",
            _md(acc),
            "### Weekly WMAPE by demand class",
            _md(cls),
            f"### Inventory simulation ({cfg['inventory']['service_level']:.0%} target service level)",
            _md(inv),
        ]
    )


def run_report(spark: SparkSession, cfg: dict[str, Any]) -> None:
    text = build_report(spark, cfg)
    print(text)
    out_dir = cfg["storage"].get("report_dir")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "summary.md"), "w") as f:
            f.write(text + "\n")
