import os
from pathlib import Path

import pandas as pd
import pytest

# Real checkpoints and eval data; tests that need them skip when the variables are unset.
MODELS_DIR = os.environ.get("ABLM_EVAL_TEST_MODELS")
EVAL_DATA = os.environ.get("ABLM_EVAL_TEST_DATA")
PAIRED_COLUMNS = ["sequence_aa:0", "sequence_aa:1"]  # heavy, light


@pytest.fixture(scope="session")
def models_dir() -> Path:
    if not MODELS_DIR:
        pytest.skip("set ABLM_EVAL_TEST_MODELS to a directory of real checkpoints")
    return Path(MODELS_DIR)


@pytest.fixture(scope="session")
def eval_sequences() -> list[str]:
    if not EVAL_DATA:
        pytest.skip("set ABLM_EVAL_TEST_DATA to a paired eval parquet (sequence_aa:0/:1)")
    df = pd.read_parquet(EVAL_DATA, columns=PAIRED_COLUMNS).head(32)
    return (df[PAIRED_COLUMNS[0]] + "<cls>" + df[PAIRED_COLUMNS[1]]).tolist()
