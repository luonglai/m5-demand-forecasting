<!-- Output of the `report` task, Databricks Free Edition job run, all 10 stores (FOODS), 22 Oct 2026 -->
## Test fold (days 1914-1941)

### Accuracy and bias, item x store

| model | wmape_daily | wmape_weekly | bias_weekly | fva_pts_weekly |
|---|---|---|---|---|
| naive | 78.7% | 39.1% | -9.0% | +0.0 |
| ets | 65.6% | 36.5% | -7.1% | +2.6 |
| croston_sba | 68.2% | 37.0% | -5.0% | +2.1 |
| stat_baseline | 67.8% | 37.8% | -4.8% | +1.3 |
| lightgbm | 65.7% | 35.6% | -6.2% | +3.4 |

### Weekly WMAPE by demand class

| demand_class | croston_sba | ets | lightgbm | naive | stat_baseline |
|---|---|---|---|---|---|
| erratic | 35.8% | 39.0% | 36.6% | 39.2% | 39.0% |
| intermittent | 45.5% | 43.4% | 42.8% | 47.6% | 45.5% |
| lumpy | 51.5% | 48.0% | 47.9% | 52.6% | 51.5% |
| no_demand | 147.4% | 101.6% | 92.1% | 100.0% | 147.4% |
| smooth | 22.8% | 24.4% | 23.4% | 25.1% | 24.4% |

### Inventory simulation (95% target service level)

| model | fill_rate | lost_units | excess_units | avg_safety_stock_value |
|---|---|---|---|---|
| naive | 91.5% | 70,364 | 1,461,145 | 145,502 |
| ets | 91.0% | 74,654 | 1,163,796 | 113,318 |
| croston_sba | 93.0% | 58,285 | 1,209,819 | 119,055 |
| stat_baseline | 92.8% | 59,643 | 1,212,620 | 118,516 |
| lightgbm | 92.0% | 66,884 | 1,135,817 | 110,899 |
