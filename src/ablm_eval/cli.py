"""The `ablm-eval` command: run AbLM-Eval tasks on pinned models from a YAML config."""

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, NoReturn

import typer
import yaml
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .lab_models import LAB_MODELS, LabModel
from .run import _eval_model, compare_results
from .run_config import load_run_spec
from .utils.directories import create_results_dir

__all__ = ["app", "apply_separator"]

app = typer.Typer(help="Evaluate antibody language models with AbLM-Eval.", no_args_is_help=True)
console = Console(stderr=True)


def _fail(err: Exception) -> NoReturn:
    console.print(f"[red]error:[/red] {escape(str(err))}")
    raise typer.Exit(1)


def apply_separator(tasks: list, model: LabModel) -> None:
    """Set the model's chain separator on every task.

    Args:
        tasks: Task configs to update in place.
        model: The model about to be evaluated. A model trained on single
            chains (``separator`` of ``None``) leaves the tasks unchanged.
    """
    if model.separator is None:
        return
    for task in tasks:
        task.separator = model.separator


@app.command()
def run(
    config: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="The run's YAML config.")
    ],
    cache_dir: Annotated[Path, typer.Option(help="Where downloaded models are kept.")] = Path(
        ".ablm-eval-cache"
    ),
    overwrite: Annotated[
        bool, typer.Option(help="Reuse task directories that already hold files.")
    ] = False,
) -> None:
    """Run every task in CONFIG on every model in it."""
    try:
        spec = load_run_spec(config)
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as err:
        _fail(err)
    try:
        create_results_dir(str(spec.output_dir), spec.tasks, overwrite)
    except Exception as err:  # noqa: BLE001 - create_results_dir raises a bare Exception
        _fail(err)

    (spec.output_dir / "models.json").write_text(
        json.dumps([asdict(m) for m in spec.models], indent=2)
    )
    separators = {m.name: m.separator for m in spec.models}
    for task in spec.tasks:
        record_path = Path(task.output_dir) / "config.json"
        record = json.loads(record_path.read_text())
        record["separator"] = separators
        record_path.write_text(json.dumps(record, indent=2))

    for model in spec.models:
        console.print(f"[bold]{escape(model.name)}[/bold]: {model.repo} @ {model.revision[:12]}")
        local = snapshot_download(model.repo, revision=model.revision, cache_dir=cache_dir)
        tasks = copy.deepcopy(spec.tasks)
        apply_separator(tasks, model)
        _eval_model(model.name, local, tasks)

    if len(spec.models) > 1:
        compare_results(configs=spec.tasks)


@app.command()
def models() -> None:
    """List the lab models a config can name."""
    table = Table("name", "repository", "revision", "separator")
    for m in LAB_MODELS.values():
        table.add_row(m.name, m.repo, m.revision[:12], m.separator or "single chains only")
    Console().print(table)
