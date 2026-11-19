"""The `ablm-eval` command: run AbLM-Eval tasks on pinned models from a YAML config."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.table import Table

from .lab_models import LAB_MODELS, LabModel
from .run import _eval_model, compare_results
from .run_config import load_run_spec
from .utils.directories import create_results_dir

__all__ = ["app", "apply_separator"]

app = typer.Typer(help="Evaluate antibody language models with AbLM-Eval.", no_args_is_help=True)
console = Console(stderr=True)


def _fail(err: Exception) -> NoReturn:
    console.print(f"[red]error:[/red] {err}")
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
) -> None:
    """Run every task in CONFIG on every model in it."""
    try:
        spec = load_run_spec(config)
    except ValueError as err:
        _fail(err)
    try:
        create_results_dir(str(spec.output_dir), spec.tasks, False)
    except Exception as err:  # noqa: BLE001 - create_results_dir raises a bare Exception
        _fail(err)

    for model in spec.models:
        console.print(f"[bold]{model.name}[/bold]: {model.repo} @ {model.revision[:12]}")
        local = snapshot_download(model.repo, revision=model.revision, cache_dir=cache_dir)
        apply_separator(spec.tasks, model)
        _eval_model(model.name, local, spec.tasks)

    if len(spec.models) > 1:
        compare_results(configs=spec.tasks)
    (spec.output_dir / "models.json").write_text(
        json.dumps([asdict(m) for m in spec.models], indent=2)
    )


@app.command()
def models() -> None:
    """List the lab models a config can name."""
    table = Table("name", "repository", "revision", "separator")
    for m in LAB_MODELS.values():
        table.add_row(m.name, m.repo, m.revision[:12], m.separator or "single chains only")
    Console().print(table)
