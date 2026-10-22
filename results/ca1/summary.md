## Test fold (days 1914-1941)

### Accuracy and bias, item x store

| model | wmape_daily | wmape_weekly | bias_weekly | fva_pts_weekly |
|---|---|---|---|---|
| naive | 78.1% | 38.6% | -7.2% | +0.0 |
| ets | 65.1% | 36.6% | -7.6% | +2.0 |
| croston_sba | 69.2% | 37.3% | -5.7% | +1.3 |
| stat_baseline | 68.1% | 38.1% | -5.7% | +0.5 |
| lightgbm | 64.8% | 36.1% | -11.3% | +2.5 |

### Weekly WMAPE by demand class

| demand_class | croston_sba | ets | lightgbm | naive | stat_baseline |
|---|---|---|---|---|---|
| erratic | 44.3% | 42.5% | 46.8% | 44.2% | 42.5% |
| intermittent | 47.3% | 45.1% | 44.9% | 47.8% | 47.3% |
| lumpy | 53.6% | 48.4% | 50.9% | 52.8% | 53.6% |
| no_demand | 56.7% | 98.6% | 75.7% | 100.0% | 56.7% |
| smooth | 21.5% | 23.7% | 21.4% | 24.3% | 23.7% |

### Inventory simulation (95% target service level)

| model | fill_rate | lost_units | excess_units | avg_safety_stock_value |
|---|---|---|---|---|
| naive | 91.7% | 7,513 | 158,077 | 15,798 |
| ets | 90.9% | 8,243 | 123,856 | 12,296 |
| croston_sba | 92.7% | 6,605 | 132,374 | 13,361 |
| stat_baseline | 92.8% | 6,566 | 131,222 | 13,211 |
| lightgbm | 91.0% | 8,172 | 117,098 | 12,122 |
