from pathlib import Path

import pytest
import yaml
from transformers import TrainingArguments

from ablm_eval import ClassificationConfig, DatasetColumns, PerPositionConfig
from ablm_eval.run_config import load_run_spec

PER_POSITION = {
    "task": "per_position",
    "dataset_name": "paired-test",
    "antibody_datatype": "paired",
    "data_path": "data/paired-test.parquet",
}


def write(tmp_path: Path, spec: dict) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def spec_with(models: list, tasks: list) -> dict:
    return {"output_dir": "results", "models": models, "tasks": tasks}


def test_loads_models_and_tasks(tmp_path):
    spec = load_run_spec(write(tmp_path, spec_with(["currab"], [PER_POSITION])))
    assert spec.output_dir == Path("results")
    assert [m.name for m in spec.models] == ["currab"]
    (task,) = spec.tasks
    assert isinstance(task, PerPositionConfig)
    assert task.dataset_name == "paired-test"
    assert task.antibody_datatype == "paired"
    assert task.data_path == "data/paired-test.parquet"


def test_dataset_columns_become_the_dataclass(tmp_path):
    task = {**PER_POSITION, "dataset_columns": {"chain_columns": ["sequence"]}}
    (loaded,) = load_run_spec(write(tmp_path, spec_with(["currab"], [task]))).tasks
    assert isinstance(loaded.dataset_columns, DatasetColumns)
    assert loaded.dataset_columns.chain_columns == ["sequence"]


def test_classification_training_args_become_the_dataclass(tmp_path):
    task = {
        "task": "classification",
        "dataset_name": "HD-CoV",
        "antibody_datatype": "paired",
        "data_path": {0: {"train": "train0.csv", "test": "test0.csv"}},
        "num_folds": 1,
        "training_args": {"output_dir": "", "learning_rate": 5e-5, "report_to": "none"},
    }
    (loaded,) = load_run_spec(write(tmp_path, spec_with(["currab"], [task]))).tasks
    assert isinstance(loaded, ClassificationConfig)
    assert isinstance(loaded.training_args, TrainingArguments)
    assert loaded.training_args.learning_rate == 5e-5


def test_unknown_task_is_refused(tmp_path):
    task = {**PER_POSITION, "task": "perposition"}
    with pytest.raises(ValueError, match="perposition"):
        load_run_spec(write(tmp_path, spec_with(["currab"], [task])))


def test_task_separator_with_model_separator_is_refused(tmp_path):
    task = {**PER_POSITION, "separator": "<cls>"}
    with pytest.raises(ValueError, match="paired-test"):
        load_run_spec(write(tmp_path, spec_with(["currab"], [task])))


def test_paired_task_with_unpaired_model_is_refused(tmp_path):
    path = write(tmp_path, spec_with(["balm-unpaired"], [PER_POSITION]))
    with pytest.raises(ValueError, match="balm-unpaired.*paired-test|paired-test.*balm-unpaired"):
        load_run_spec(path)


def test_unpaired_task_with_unpaired_model_loads(tmp_path):
    task = {**PER_POSITION, "antibody_datatype": "unpaired"}
    spec = load_run_spec(write(tmp_path, spec_with(["balm-unpaired"], [task])))
    assert spec.tasks[0].antibody_datatype == "unpaired"


def test_duplicate_model_names_are_refused(tmp_path):
    with pytest.raises(ValueError, match="currab"):
        load_run_spec(write(tmp_path, spec_with(["currab", "currab"], [PER_POSITION])))


def test_two_tasks_with_one_output_directory_are_refused(tmp_path):
    with pytest.raises(ValueError, match="paired-test_per_pos_inference"):
        load_run_spec(write(tmp_path, spec_with(["currab"], [PER_POSITION, PER_POSITION])))
