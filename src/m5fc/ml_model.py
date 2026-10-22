"""Global LightGBM model across all series ("one model for every SKU").

Features are built in Spark (gold layer); training runs on the driver. All lag
and rolling features look back at least `horizon` days, so a single model
forecasts the whole 28-day horizon directly, without feeding forecasts back in.
Tweedie loss suits retail demand: many zeros and a long right tail.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from m5fc.io import read_table, write_table
from m5fc.transform import FEATURE_LAGS

CATEGORICAL = ["item_id", "dept_id", "store_id", "event_type"]
NUMERIC = (
    [f"lag_{k}" for k in FEATURE_LAGS]
    + ["rmean_7_lag28", "rmean_28_lag28", "rstd_28_lag28", "rmean_91_lag28"]
    + ["sell_price", "price_rel", "price_change_7", "days_since_launch"]
    + ["wday", "month", "is_event", "snap"]
)
FEATURES = CATEGORICAL + NUMERIC


def _to_pandas(df: DataFrame, categories: dict[str, list[str]] | None = None) -> pd.DataFrame:
    pdf = df.select("id", "d_num", "sales", *FEATURES).toPandas()
    for c in CATEGORICAL:
        cats = categories[c] if categories else sorted(pdf[c].dropna().unique())
        pdf[c] = pd.Categorical(pdf[c], categories=cats)
    for c in NUMERIC:
        pdf[c] = pdf[c].astype("float32")
    return pdf


def train_and_forecast(gold: DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    import lightgbm as lgb

    h, train_days = cfg["forecast"]["horizon"], cfg["forecast"]["train_days"]
    params = dict(cfg["lightgbm"])
    rounds = params.pop("num_boost_round")
    forecasts, importances = [], []
    for cutoff in cfg["forecast"]["cutoffs"]:
        train_sdf = gold.filter(
            (F.col("d_num") <= cutoff) & (F.col("d_num") > cutoff - train_days) & F.col("rmean_91_lag28").isNotNull()
        )
        train = _to_pandas(train_sdf)
        cats = {c: list(train[c].cat.categories) for c in CATEGORICAL}
        test = _to_pandas(gold.filter((F.col("d_num") > cutoff) & (F.col("d_num") <= cutoff + h)), cats)

        booster = lgb.train(
            params,
            lgb.Dataset(train[FEATURES], label=train["sales"], categorical_feature=CATEGORICAL),
            num_boost_round=rounds,
        )
        out = test[["id", "d_num"]].copy()
        out["forecast"] = booster.predict(test[FEATURES]).clip(min=0)
        out["model"] = "lightgbm"
        out["cutoff"] = cutoff
        forecasts.append(out)
        importances.append(
            pd.DataFrame(
                {
                    "cutoff": cutoff,
                    "feature": FEATURES,
                    "gain": booster.feature_importance(importance_type="gain"),
                }
            )
        )
    fc = pd.concat(forecasts, ignore_index=True)[["id", "cutoff", "d_num", "model", "forecast"]]
    fc["d_num"] = fc["d_num"].astype("int32")
    fc["cutoff"] = fc["cutoff"].astype("int32")
    fc["forecast"] = fc["forecast"].astype(float)
    return fc, pd.concat(importances, ignore_index=True)


def run_ml(spark: SparkSession, cfg: dict[str, Any]) -> None:
    fc, imp = train_and_forecast(read_table(spark, cfg, "gold_features"), cfg)
    write_table(spark.createDataFrame(fc), cfg, "forecast_lightgbm")
    write_table(spark.createDataFrame(imp), cfg, "feature_importance")
