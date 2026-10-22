"""Statistical baseline forecasts, fitted per series in parallel with Spark.

Each Spark task receives all series of one store x department as a pandas
frame (`applyInPandas`) and fits them with Nixtla's statsforecast. This is the
standard "many models" pattern for forecasting thousands of SKUs on Spark.

Models
  naive          seasonal naive (same weekday last week): the benchmark for FVA
  ets            AutoETS with weekly seasonality, for regular sellers
  croston_sba    Croston with Syntetos-Boylan bias correction, for intermittent items
  stat_baseline  per-series choice between ets and croston_sba by demand class
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from m5fc.io import read_table, write_table

ADI_CUTOFF = 1.32  # Syntetos-Boylan (2005) thresholds
CV2_CUTOFF = 0.49
CLASS_HISTORY_DAYS = 365
FORECAST_SCHEMA = "id string, cutoff int, d_num int, model string, forecast double"
CLASS_SCHEMA = "id string, cutoff int, adi double, cv2 double, demand_class string"


def classify_demand(y: np.ndarray) -> tuple[float, float, str]:
    """Syntetos-Boylan demand classification of one history."""
    nonzero = y[y > 0]
    if len(nonzero) == 0:
        return float("inf"), float("nan"), "no_demand"
    adi = len(y) / len(nonzero)
    cv2 = float((nonzero.std() / nonzero.mean()) ** 2) if len(nonzero) > 1 else 0.0
    if adi < ADI_CUTOFF:
        cls = "smooth" if cv2 < CV2_CUTOFF else "erratic"
    else:
        cls = "intermittent" if cv2 < CV2_CUTOFF else "lumpy"
    return float(adi), cv2, cls


def _forecast_group(pdf: pd.DataFrame, cutoff: int, horizon: int, season: int) -> pd.DataFrame:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, CrostonSBA, Naive, SeasonalNaive

    df = pdf.rename(columns={"id": "unique_id", "d_num": "ds", "sales": "y"})[["unique_id", "ds", "y"]]
    df = df.sort_values(["unique_id", "ds"])
    sf = StatsForecast(
        models=[
            SeasonalNaive(season_length=season, alias="naive"),
            AutoETS(season_length=season, alias="ets"),
            CrostonSBA(alias="croston_sba"),
        ],
        freq=1,
        n_jobs=1,
        fallback_model=Naive(),
    )
    fc = sf.forecast(df=df, h=horizon)
    if "unique_id" not in fc.columns:  # older statsforecast versions return it as the index
        fc = fc.reset_index()
    long = fc.melt(
        id_vars=["unique_id", "ds"], value_vars=["naive", "ets", "croston_sba"], var_name="model", value_name="forecast"
    )

    # Choose the baseline per series from its demand class.
    classes = {uid: classify_demand(g["y"].to_numpy()[-CLASS_HISTORY_DAYS:])[2] for uid, g in df.groupby("unique_id")}
    pick = fc[["unique_id", "ds"]].copy()
    use_croston = fc["unique_id"].map(lambda u: classes[u] in ("intermittent", "lumpy", "no_demand"))
    pick["forecast"] = np.where(use_croston, fc["croston_sba"], fc["ets"])
    pick["model"] = "stat_baseline"
    long = pd.concat([long, pick[long.columns]], ignore_index=True)

    long["forecast"] = long["forecast"].astype(float).clip(lower=0)
    long["cutoff"] = cutoff
    return long.rename(columns={"unique_id": "id", "ds": "d_num"})[["id", "cutoff", "d_num", "model", "forecast"]]


def _classes_group(pdf: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    rows = []
    for uid, g in pdf.sort_values("d_num").groupby("id"):
        adi, cv2, cls = classify_demand(g["sales"].to_numpy()[-CLASS_HISTORY_DAYS:])
        rows.append((uid, cutoff, adi, cv2, cls))
    return pd.DataFrame(rows, columns=["id", "cutoff", "adi", "cv2", "demand_class"])


def forecast_baseline(silver: DataFrame, cfg: dict[str, Any]) -> tuple[DataFrame, DataFrame]:
    h, season = cfg["forecast"]["horizon"], cfg["forecast"]["season_length"]
    fcs, classes = [], []
    for cutoff in cfg["forecast"]["cutoffs"]:
        hist = silver.filter(F.col("d_num") <= cutoff).select("id", "store_id", "dept_id", "d_num", "sales")
        grouped = hist.groupBy("store_id", "dept_id")
        # partial(), not a lambda with defaults: Spark treats a 2-argument function as (key, pdf)
        fcs.append(
            grouped.applyInPandas(
                partial(_forecast_group, cutoff=cutoff, horizon=h, season=season), schema=FORECAST_SCHEMA
            )
        )
        classes.append(grouped.applyInPandas(partial(_classes_group, cutoff=cutoff), schema=CLASS_SCHEMA))
    union = fcs[0]
    for f in fcs[1:]:
        union = union.unionByName(f)
    cls = classes[0]
    for c in classes[1:]:
        cls = cls.unionByName(c)
    return union, cls


def run_baseline(spark: SparkSession, cfg: dict[str, Any]) -> None:
    fc, cls = forecast_baseline(read_table(spark, cfg, "silver_sales"), cfg)
    write_table(fc, cfg, "forecast_baseline")
    write_table(cls, cfg, "demand_class")
