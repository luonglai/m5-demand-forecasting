# Setup guide

Work through these in order. Steps 1–3 need no cloud account.

## 1. Get the data

1. Sign in at kaggle.com and open the [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition.
2. Accept the competition rules (Data tab), then download the files.
3. Put `calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` in `data/raw/`. The other files are not needed.

## 2. Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

For a first real-data run, use the one-store config (CA_1, about 10-15 minutes):

```bash
m5fc --config conf/local_ca1.yml --stage all
cat results/ca1/summary.md
```

`conf/local.yml` runs all 10 stores. LightGBM trains
on the driver, so the full run needs roughly 8–12 GB of RAM. If memory is
tight, reduce `train_days` or keep fewer stores.

Each stage can be re-run on its own, for example `--stage ml`.

## 3. Push to GitHub

Create a public repository called `m5-demand-forecasting` and push with GitHub
Desktop. The CI workflow runs on the first push: check the **Actions** tab.
It needs no secrets; the bundle validation step skips itself until step 5.

## 4. Databricks Free Edition

1. Sign up for [Databricks Free Edition](https://www.databricks.com/learn/free-edition) with a personal email.
2. Install the [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install) and log in:
   ```bash
   databricks auth login --host https://<your-workspace>.cloud.databricks.com
   ```
3. Create the schema and the volume for raw files (or run the job's `setup` task once):
   ```sql
   CREATE SCHEMA IF NOT EXISTS workspace.m5;
   CREATE VOLUME IF NOT EXISTS workspace.m5.raw;
   ```
4. Upload the three CSVs to the volume:
   ```bash
   databricks fs cp data/raw/calendar.csv dbfs:/Volumes/workspace/m5/raw/
   databricks fs cp data/raw/sell_prices.csv dbfs:/Volumes/workspace/m5/raw/
   databricks fs cp data/raw/sales_train_evaluation.csv dbfs:/Volumes/workspace/m5/raw/
   ```
5. Deploy and run:
   ```bash
   databricks bundle validate -t dev
   databricks bundle deploy -t dev
   databricks bundle run m5_forecasting -t dev
   ```
6. Open **Jobs & Pipelines** in the workspace to see the task graph, run times and logs. Screenshot the successful run for the README.

**Things to check on the first deploy:**
- The job's serverless environment installs the wheel and its dependencies (statsforecast, LightGBM) from PyPI. If that fails, check the error, try `environment_version: "2"` or `"3"` in `resources/m5_forecasting_job.yml`, and check whether Free Edition's restricted internet access blocks PyPI.
- Serverless can keep an old wheel cached. Bump `version` in `pyproject.toml` when you change code and redeploy.
- If the `ml` task runs out of memory, set `scope.stores` or lower `forecast.train_days` in `conf/databricks_run.yml`.
- Free Edition allows at most 5 concurrent job tasks and has a daily compute quota.

## 5. CI/CD to Databricks

1. In Databricks: **Settings → Developer → Access tokens → Generate new token**.
2. In GitHub: **Settings → Secrets and variables → Actions**, add `DATABRICKS_HOST` (workspace URL) and `DATABRICKS_TOKEN`.
3. Pushes to `main` now validate the bundle in CI and deploy it after CI passes. To deploy and run by hand: **Actions → Deploy to Databricks → Run workflow**.

## 6. Azure Databricks (optional, small cost risk)

Free Edition runs on Databricks' own infrastructure. To deploy the same job to
**Azure Databricks**:

1. Create an Azure account and an **Azure Databricks** workspace (Premium tier) in the Azure portal. Check the current trial terms and set a **budget alert** first: an Azure subscription needs a payment card, and anything beyond the trial credit is billed.
2. Put the workspace URL in `databricks.yml` (`variables.azure_host`), or pass it with `--var azure_host=...`.
3. Upload the CSVs to a volume as in step 4, then:
   ```bash
   databricks bundle deploy -t azure
   databricks bundle run m5_forecasting -t azure
   ```
4. For CD, add `AZURE_DATABRICKS_HOST` and `AZURE_DATABRICKS_TOKEN` secrets and run the deploy workflow with target `azure`.
5. When finished, **delete the resource group** so nothing keeps billing.
