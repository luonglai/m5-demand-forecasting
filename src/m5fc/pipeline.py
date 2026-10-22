"""Command-line entry point. Each stage is one Databricks job task.

python -m m5fc.pipeline --config conf/local.yml --stage all
"""

from __future__ import annotations

import argparse
import logging
import time

from m5fc.baseline import run_baseline
from m5fc.config import load_config
from m5fc.evaluate import run_evaluate
from m5fc.inventory import run_inventory
from m5fc.io import ensure_storage, get_spark
from m5fc.ml_model import run_ml
from m5fc.report import run_report
from m5fc.transform import build_bronze, build_gold, build_silver

STAGES = {
    "setup": ensure_storage,
    "bronze": build_bronze,
    "silver": build_silver,
    "gold": build_gold,
    "baseline": run_baseline,
    "ml": run_ml,
    "evaluate": run_evaluate,
    "inventory": run_inventory,
    "report": run_report,
}

log = logging.getLogger("m5fc")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="M5 demand forecasting pipeline")
    p.add_argument("--config", required=True)
    p.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(a.config)
    spark = get_spark()
    for name in STAGES if a.stage == "all" else [a.stage]:
        t0 = time.time()
        log.info("stage %s: start", name)
        STAGES[name](spark, cfg)
        log.info("stage %s: done in %.1fs", name, time.time() - t0)


if __name__ == "__main__":
    main()
