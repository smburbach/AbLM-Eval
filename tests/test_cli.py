import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from huggingface_hub import snapshot_download
from typer.testing import CliRunner

from ablm_eval import PerPositionConfig, evaluate_ablms
from ablm_eval.cli import app, apply_separator
from ablm_eval.lab_models import LAB_MODELS, resolve_model
from ablm_eval.run_config import load_run_spec

FIXTURE = Path(__file__).parent / "data" / "paired-mini.parquet"
TINY = {
    "name": "tiny",
    "path": "brineylab/8M_full_checkpoint-500000",
    "revision": "01e6532f909186a7f32ffd3e5fd6ced80daa0cc1",
    "separator": "<cls><cls>",
}
PER_POSITION = {
    "task": "per_position",
    "dataset_name": "mini",
    "antibody_datatype": "paired",
    "data_path": str(FIXTURE),
    "num_proc": 1,
}


def write_spec(tmp_path: Path, output_dir: Path, models: list) -> Path:
    path = tmp_path / "run.yaml"
    spec = {"output_dir": str(output_dir), "models": models, "tasks": [PER_POSITION]}
    path.write_text(yaml.safe_dump(spec))
    return path


def test_models_lists_the_registry():
    result = CliRunner().invoke(app, ["models"])
    assert result.exit_code == 0
    for name in LAB_MODELS:
        assert name in result.output


def test_apply_separator_sets_every_task(tmp_path):
    tasks = [
        PerPositionConfig(data_path=str(FIXTURE), antibody_datatype="paired", dataset_name=n)
        for n in ("a", "b")
    ]
    apply_separator(tasks, resolve_model("currab"))
    assert [t.separator for t in tasks] == ["<cls>", "<cls>"]


def test_nonempty_output_dir_exits_cleanly(tmp_path):
    out = tmp_path / "results"
    config = write_spec(tmp_path, out, ["currab"])
    task_dir = out / load_run_spec(config).tasks[0].task_dir
    task_dir.mkdir(parents=True)
    (task_dir / "old.txt").write_text("x")
    result = CliRunner().invoke(
        app, ["run", str(config), "--cache-dir", str(tmp_path / "cache")], env={"COLUMNS": "1000"}
    )
    assert result.exit_code == 1
    assert "not empty" in result.output
    assert "Traceback" not in result.output


@pytest.mark.slow
def test_run_matches_the_python_api(tmp_path):
    cache = tmp_path / "cache"
    cli_out = tmp_path / "cli"
    result = CliRunner().invoke(
        app, ["run", str(write_spec(tmp_path, cli_out, [TINY])), "--cache-dir", str(cache)]
    )
    assert result.exit_code == 0, result.output

    local = snapshot_download(TINY["path"], revision=TINY["revision"], cache_dir=cache)
    api_out = tmp_path / "api"
    fields = {k: v for k, v in PER_POSITION.items() if k != "task"}
    evaluate_ablms(
        {"tiny": local},
        [PerPositionConfig(**fields, separator="<cls><cls>")],
        shared_output_dir=str(api_out),
        generate_comparisons=False,
    )

    (cli_file,) = cli_out.rglob("*per-position-inference.parquet")
    (api_file,) = api_out.rglob("*per-position-inference.parquet")
    cli_df, api_df = pd.read_parquet(cli_file), pd.read_parquet(api_file)
    assert cli_df["sequence"].str.contains("<cls><cls>").all()
    pd.testing.assert_frame_equal(cli_df, api_df, check_exact=False, rtol=1e-5)

    record = json.loads((cli_out / "models.json").read_text())
    assert record == [
        {
            "name": "tiny",
            "repo": TINY["path"],
            "revision": TINY["revision"],
            "separator": "<cls><cls>",
        }
    ]


def test_each_model_gets_its_own_task_configs(tmp_path, monkeypatch):
    seen = []

    def fake_eval(model_name, model_path, configs):
        (task,) = configs
        seen.append((model_name, task.separator, task.data_path))
        task.data_path = "processed-by-" + model_name

    monkeypatch.setattr("ablm_eval.cli.snapshot_download", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr("ablm_eval.cli._eval_model", fake_eval)
    monkeypatch.setattr("ablm_eval.cli.compare_results", lambda **k: None)
    models = [{**TINY, "name": "a"}, {**TINY, "name": "b", "separator": "<cls>"}]
    config = write_spec(tmp_path, tmp_path / "results", models)
    result = CliRunner().invoke(app, ["run", str(config), "--cache-dir", str(tmp_path / "c")])
    assert result.exit_code == 0, result.output
    assert seen == [("a", "<cls><cls>", str(FIXTURE)), ("b", "<cls>", str(FIXTURE))]


def test_malformed_task_field_exits_cleanly(tmp_path):
    path = tmp_path / "run.yaml"
    task = {k: v for k, v in PER_POSITION.items() if k != "data_path"} | {"datapath": "x"}
    path.write_text(yaml.safe_dump({"output_dir": "r", "models": ["currab"], "tasks": [task]}))
    result = CliRunner().invoke(app, ["run", str(path)])
    assert result.exit_code == 1
    assert "datapath" in result.output
    assert "Traceback" not in result.output


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    """Stub the download, the evaluation and the comparison; record each model run."""
    seen = []
    monkeypatch.setattr("ablm_eval.cli.snapshot_download", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr("ablm_eval.cli._eval_model", lambda name, path, cfgs: seen.append(name))
    monkeypatch.setattr("ablm_eval.cli.compare_results", lambda **k: None)
    return seen


TWO_MODELS = [{**TINY, "name": "a"}, {**TINY, "name": "b", "separator": "<cls>"}]


def test_config_json_records_each_models_separator(tmp_path, fake_run):
    out = tmp_path / "results"
    config = write_spec(tmp_path, out, TWO_MODELS)
    assert CliRunner().invoke(app, ["run", str(config)]).exit_code == 0
    (record,) = out.rglob("config.json")
    assert json.loads(record.read_text())["separator"] == {"a": "<cls><cls>", "b": "<cls>"}


def test_overwrite_reuses_a_nonempty_output_dir(tmp_path, fake_run):
    out = tmp_path / "results"
    config = write_spec(tmp_path, out, TWO_MODELS)
    task_dir = out / load_run_spec(config).tasks[0].task_dir
    task_dir.mkdir(parents=True)
    (task_dir / "old.txt").write_text("x")
    result = CliRunner().invoke(app, ["run", str(config), "--overwrite"])
    assert result.exit_code == 0, result.output
    assert fake_run == ["a", "b"]


def test_models_json_is_written_before_any_model_runs(tmp_path, monkeypatch):
    out = tmp_path / "results"

    def crash(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr("ablm_eval.cli.snapshot_download", crash)
    CliRunner().invoke(app, ["run", str(write_spec(tmp_path, out, TWO_MODELS))])
    assert [m["name"] for m in json.loads((out / "models.json").read_text())] == ["a", "b"]


def test_error_text_keeps_square_brackets(tmp_path):
    path = tmp_path / "run.yaml"
    task = {**PER_POSITION, "task": "[bold]per_position"}
    path.write_text(yaml.safe_dump({"output_dir": "r", "models": ["currab"], "tasks": [task]}))
    result = CliRunner().invoke(app, ["run", str(path)], env={"COLUMNS": "1000"})
    assert result.exit_code == 1
    assert "[bold]per_position" in result.output
