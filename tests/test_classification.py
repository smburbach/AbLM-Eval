from pathlib import Path

import pandas as pd
import pytest
import yaml
from huggingface_hub import snapshot_download
from typer.testing import CliRunner

from ablm_eval.cli import app
from ablm_eval.lab_models import LAB_MODELS
from ablm_eval.tasks.classification.train_script import (
    _def_training_args,
    _model_kwargs,
)

DATA = Path(__file__).parent / "data"
TINY = {
    "name": "tiny",
    "path": "brineylab/8M_full_checkpoint-500000",
    "revision": "01e6532f909186a7f32ffd3e5fd6ced80daa0cc1",
    "separator": "<cls><cls>",
}
CLASSIFICATION = {
    "task": "classification",
    "dataset_name": "hdcov-mini",
    "antibody_datatype": "paired",
    "data_path": {
        0: {
            "train": str(DATA / "hdcov-mini-train.csv"),
            "test": str(DATA / "hdcov-mini-test.csv"),
        }
    },
    "num_folds": 1,
    "num_proc": 1,
}


def _config_dir(repo: str, revision: str, cache: Path) -> str:
    return snapshot_download(
        repo, revision=revision, cache_dir=cache, allow_patterns=["*.json"]
    )


def test_default_training_args_build():
    class NoTrainingArgs:
        training_args = None
        output_dir = "out"

    args = _def_training_args("run", NoTrainingArgs())
    assert args.warmup_steps == 0.1


def test_attention_classifier_is_left_out_for_a_model_without_it(tmp_path):
    path = _config_dir(TINY["path"], TINY["revision"], tmp_path)
    assert _model_kwargs(None, path) == {}


def test_attention_classifier_set_for_a_model_without_it_is_refused(tmp_path):
    path = _config_dir(TINY["path"], TINY["revision"], tmp_path)
    with pytest.raises(ValueError, match="attention_classifier"):
        _model_kwargs(True, path)


@pytest.mark.parametrize(("given", "passed"), [(None, True), (False, False)])
def test_attention_classifier_reaches_balm_moe(tmp_path, given, passed):
    moe = LAB_MODELS["balm-moe"]
    path = _config_dir(moe.repo, moe.revision, tmp_path)
    assert _model_kwargs(given, path) == {"attention_classifier": passed}


@pytest.mark.slow
def test_classification_trains_and_scores(tmp_path):
    out = tmp_path / "out"
    spec = {"output_dir": str(out), "models": [TINY], "tasks": [CLASSIFICATION]}
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(spec))

    result = CliRunner().invoke(
        app, ["run", str(config), "--cache-dir", str(tmp_path / "cache")]
    )
    assert result.exit_code == 0, result.output

    (results,) = out.rglob("*classification.csv")
    scores = pd.read_csv(results)
    assert len(scores) == 1
    assert scores["model"].tolist() == ["tiny"]
