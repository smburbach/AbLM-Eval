import math
import os

import pandas as pd
import pytest
import torch

from ablm_eval import DatasetColumns, InferenceConfig
from ablm_eval.tasks.inference.inference_run import run_inference

# Real checkpoints and eval data; skipped when the variables are unset.
MODELS_DIR = os.environ.get("ABLM_EVAL_TEST_MODELS")
EVAL_DATA = os.environ.get("ABLM_EVAL_TEST_DATA")

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR and EVAL_DATA and torch.cuda.is_available()),
    reason="needs ABLM_EVAL_TEST_MODELS, ABLM_EVAL_TEST_DATA and a GPU",
)


@pytest.mark.parametrize("name", ["sota_baseline", "balm_paired"])  # EsmForMaskedLM, Roberta
def test_eval_loss_matches_perplexity(name, tmp_path):
    data = tmp_path / "eval.parquet"
    pd.read_parquet(EVAL_DATA).head(512).to_parquet(data)
    (tmp_path / "results").mkdir()
    config = InferenceConfig(
        data_path=str(data),
        antibody_datatype="paired",
        dataset_columns=DatasetColumns(chain_columns=["sequence_aa:0", "sequence_aa:1"]),
        dataset_name="eval",
        output_dir=str(tmp_path),
        num_proc=4,
    )
    run_inference(name, os.path.join(MODELS_DIR, name), config)

    results = pd.read_csv(tmp_path / "results" / f"{name}_eval-inference.csv").iloc[0]
    # eval_loss averages per-batch means; perplexity pools tokens, so allow a small gap
    assert results["eval_loss"] == pytest.approx(math.log(results["eval_perplexity"]), rel=0.05)
