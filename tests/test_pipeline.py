"""End-to-end smoke test: every stage runs on synthetic data and the outputs make sense."""

import pytest
from pyspark.sql import functions as F

from m5fc.io import read_table
from m5fc.pipeline import STAGES


@pytest.fixture(scope="module")
def ran(spark, cfg):
    for stage in STAGES.values():
        stage(spark, cfg)
    return True


def test_forecasts_cover_every_series_horizon_and_fold(spark, cfg, ran):
    silver = read_table(spark, cfg, "silver_sales")
    n_series = silver.select("id").distinct().count()
    h, folds = cfg["forecast"]["horizon"], len(cfg["forecast"]["cutoffs"])
    base = read_table(spark, cfg, "forecast_baseline")
    assert base.select("model").distinct().count() == 4
    assert base.count() == n_series * h * folds * 4
    ml = read_table(spark, cfg, "forecast_lightgbm")
    assert ml.count() == n_series * h * folds


def test_forecasts_are_non_negative_and_after_cutoff(spark, cfg, ran):
    for name in ["forecast_baseline", "forecast_lightgbm"]:
        fc = read_table(spark, cfg, name)
        assert fc.filter(F.col("forecast") < 0).count() == 0
        assert fc.filter(F.col("d_num") <= F.col("cutoff")).count() == 0


def test_naive_has_zero_fva_and_metrics_are_finite(spark, cfg, ran):
    m = read_table(spark, cfg, "forecast_metrics")
    naive = m.filter(F.col("model") == "naive").agg(F.max(F.abs("fva_pts"))).first()[0]
    assert naive == pytest.approx(0.0)
    total = m.filter((F.col("level") == "total") & (F.col("grain") == "weekly")).collect()
    assert all(0 <= r.wmape < 5 for r in total)


def test_inventory_simulation_outputs(spark, cfg, ran):
    inv = read_table(spark, cfg, "inventory_simulation").filter(F.col("level") == "total").collect()
    assert {r.model for r in inv} == {"naive", "ets", "croston_sba", "stat_baseline", "lightgbm"}
    assert all(0 < r.fill_rate <= 1 for r in inv)
