import copy
import os

import pytest

from m5fc.config import load_config
from m5fc.io import get_spark
from m5fc.synthetic import write_synthetic_m5

ROOT = os.path.dirname(os.path.dirname(__file__))
N_DAYS = 300


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("SPARK_SHUFFLE_PARTITIONS", "4")
    s = get_spark("m5fc-tests")
    yield s
    s.stop()


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    base = tmp_path_factory.mktemp("m5")
    raw = base / "raw"
    write_synthetic_m5(str(raw), n_days=N_DAYS, items_per_dept=4)
    c = copy.deepcopy(load_config(os.path.join(ROOT, "conf", "synthetic.yml")))
    c["storage"]["raw_path"] = str(raw)
    c["storage"]["base_path"] = str(base / "lake")
    c["storage"]["report_dir"] = str(base / "results")
    c["forecast"]["cutoffs"] = [N_DAYS - 56, N_DAYS - 28]
    c["forecast"]["train_days"] = 150
    c["lightgbm"]["num_boost_round"] = 30
    return c
