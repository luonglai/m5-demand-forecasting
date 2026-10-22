from pyspark.sql import functions as F

from m5fc.io import read_raw_csv
from m5fc.transform import add_features, build_silver_frame, melt_sales


def _silver(spark, cfg):
    return build_silver_frame(
        read_raw_csv(spark, cfg, "sales_train_evaluation.csv"),
        read_raw_csv(spark, cfg, "calendar.csv"),
        read_raw_csv(spark, cfg, "sell_prices.csv"),
        cfg,
    )


def test_melt_keeps_every_series_day(spark, cfg):
    wide = read_raw_csv(spark, cfg, "sales_train_evaluation.csv")
    n_days = len([c for c in wide.columns if c.startswith("d_")])
    assert melt_sales(wide).count() == wide.count() * n_days


def test_silver_scope_and_prelaunch_filter(spark, cfg):
    silver = _silver(spark, cfg)
    assert {r.cat_id for r in silver.select("cat_id").distinct().collect()} == {"FOODS"}
    assert silver.filter(F.col("sell_price").isNull()).count() == 0
    # items launched late must start on their launch day, not on d_1
    first_days = silver.groupBy("id").agg(F.min("d_num").alias("first")).collect()
    assert any(r.first > 1 for r in first_days)


def test_silver_has_one_row_per_series_day(spark, cfg):
    silver = _silver(spark, cfg)
    assert silver.count() == silver.select("id", "d_num").distinct().count()


def test_lag_features_look_back_exactly_k_days(spark, cfg):
    gold = add_features(_silver(spark, cfg))
    row = gold.filter(F.col("lag_28").isNotNull()).orderBy("id", "d_num").first()
    past = gold.filter((F.col("id") == row.id) & (F.col("d_num") == row.d_num - 28)).first()
    assert row.lag_28 == past.sales


def test_rolling_features_never_use_the_horizon(spark, cfg):
    """rmean_7_lag28 at day t must equal the mean of days t-34..t-28 only."""
    gold = add_features(_silver(spark, cfg)).cache()
    row = gold.filter(F.col("d_num") == 200).orderBy("id").first()
    window = gold.filter((F.col("id") == row.id) & F.col("d_num").between(200 - 34, 200 - 28))
    expected = window.agg(F.avg("sales")).first()[0]
    assert abs(row.rmean_7_lag28 - expected) < 1e-9
