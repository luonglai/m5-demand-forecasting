"""Generate small synthetic files with exactly the M5 layout.

Used by the unit tests and CI so the pipeline can run without the Kaggle data.
It mimics the M5 quirks that matter for the pipeline: wide `d_1..d_N` sales
columns, items that launch part-way through (zeros and no price before launch),
intermittent sellers, weekly seasonality, SNAP days, events and price changes.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

WEEKDAYS = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def make_calendar(n_days: int, start: date = date(2011, 1, 29)) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_days):
        dt = start + timedelta(days=i)
        event = rng.random() < 0.03
        rows.append(
            {
                "date": dt.isoformat(),
                "wm_yr_wk": 11101 + i // 7,
                "weekday": WEEKDAYS[i % 7],
                "wday": i % 7 + 1,
                "month": dt.month,
                "year": dt.year,
                "d": f"d_{i + 1}",
                "event_name_1": "SyntheticEvent" if event else None,
                "event_type_1": "Cultural" if event else None,
                "event_name_2": None,
                "event_type_2": None,
                "snap_CA": int(dt.day <= 10),
                "snap_TX": int(dt.day in (1, 3, 5, 6, 7, 9, 11, 12, 13, 15)),
                "snap_WI": int(dt.day in (2, 3, 5, 6, 8, 9, 11, 12, 14, 15)),
            }
        )
    return pd.DataFrame(rows)


def make_sales_and_prices(
    n_days: int,
    stores: tuple[str, ...] = ("CA_1", "TX_1"),
    depts: tuple[str, ...] = ("FOODS_1", "FOODS_2", "HOBBIES_1"),
    items_per_dept: int = 6,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_weeks = (n_days + 6) // 7
    weekly = np.array([1.25, 1.3, 0.95, 0.85, 0.85, 0.9, 1.0])  # Sat..Fri
    sales_rows, price_rows = [], []
    for dept in depts:
        cat = dept.rsplit("_", 1)[0]
        for k in range(1, items_per_dept + 1):
            item = f"{dept}_{k:03d}"
            base_price = round(float(rng.uniform(1, 12)), 2)
            for store in stores:
                state = store.split("_")[0]
                intermittent = k % 3 == 0
                level = rng.uniform(0.15, 0.5) if intermittent else rng.uniform(2, 15)
                launch_day = int(rng.integers(0, n_days // 3)) if k % 4 == 0 else 0
                launch_week = launch_day // 7
                # weekly prices, with occasional 4-week promotions
                prices = np.full(n_weeks, base_price)
                for _ in range(max(1, n_weeks // 26)):
                    w = int(rng.integers(0, n_weeks))
                    prices[w : w + 4] = round(base_price * 0.8, 2)
                for w in range(launch_week, n_weeks):
                    price_rows.append(
                        {"store_id": store, "item_id": item, "wm_yr_wk": 11101 + w, "sell_price": float(prices[w])}
                    )
                t = np.arange(n_days)
                promo = (prices[t // 7] < base_price).astype(float)
                lam = level * weekly[t % 7] * (1 + 0.3 * promo) * (1 + 0.0004 * t)
                y = rng.poisson(lam)
                y[:launch_day] = 0
                sales_rows.append(
                    {
                        "id": f"{item}_{store}_evaluation",
                        "item_id": item,
                        "dept_id": dept,
                        "cat_id": cat,
                        "store_id": store,
                        "state_id": state,
                        **{f"d_{i + 1}": int(v) for i, v in enumerate(y)},
                    }
                )
    return pd.DataFrame(sales_rows), pd.DataFrame(price_rows)


def write_synthetic_m5(out_dir: str, n_days: int = 420, **kwargs) -> None:
    os.makedirs(out_dir, exist_ok=True)
    make_calendar(n_days + 28).to_csv(os.path.join(out_dir, "calendar.csv"), index=False)
    sales, prices = make_sales_and_prices(n_days, **kwargs)
    sales.to_csv(os.path.join(out_dir, "sales_train_evaluation.csv"), index=False)
    prices.to_csv(os.path.join(out_dir, "sell_prices.csv"), index=False)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Write synthetic M5-shaped CSVs")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--days", type=int, default=420)
    a = p.parse_args()
    write_synthetic_m5(a.out, a.days)
    print(f"Synthetic M5 files written to {a.out}")
