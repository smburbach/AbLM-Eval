"""Read a YAML run config into models and task configs, refusing ones that cannot run."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from transformers import TrainingArguments

from .lab_models import LabModel, resolve_model
from .tasks import (
    ClassificationConfig,
    InferenceConfig,
    MutationPredConfig,
    NaturalnessConfig,
    PerPositionConfig,
    RoutingConfig,
)
from .utils import DatasetColumns

__all__ = ["TASKS", "RunSpec", "load_run_spec"]

TASKS: dict[str, type] = {
    "inference": InferenceConfig,
    "per_position": PerPositionConfig,
    "mutation_prediction": MutationPredConfig,
    "naturalness": NaturalnessConfig,
    "classification": ClassificationConfig,
    "routing": RoutingConfig,
}


@dataclass
class RunSpec:
    """What one `ablm-eval run` evaluates.

    Attributes:
        output_dir: Directory the task results are written under.
        models: The pinned models to evaluate, in config order.
        tasks: Task configs, built from the existing config classes.
    """

    output_dir: Path
    models: list[LabModel]
    tasks: list


def load_run_spec(path: Path) -> RunSpec:
    """Read and check a run config.

    Args:
        path: The YAML file, with ``output_dir``, ``models`` and ``tasks``.

    Returns:
        The models and task configs to run.

    Raises:
        ValueError: On an unknown task, a duplicate model name, a task that
            sets ``separator`` while a model has its own, or a paired task
            with a model trained on single chains only.
    """
    raw = yaml.safe_load(Path(path).read_text())
    models = [resolve_model(entry) for entry in raw["models"]]
    duplicates = [name for name, n in Counter(m.name for m in models).items() if n > 1]
    if duplicates:
        raise ValueError(f"model names appear more than once: {', '.join(duplicates)}")

    tasks = []
    for entry in raw["tasks"]:
        fields = dict(entry)
        kind = fields.pop("task")
        if kind not in TASKS:
            raise ValueError(f"unknown task {kind!r}; tasks: {', '.join(TASKS)}")
        name = fields.get("dataset_name", kind)
        if "separator" in fields and any(m.separator is not None for m in models):
            raise ValueError(
                f"task {name!r} sets separator, but each model brings its own; remove it"
            )
        if fields.get("antibody_datatype") == "paired":
            unpaired = [m.name for m in models if m.separator is None]
            if unpaired:
                raise ValueError(
                    f"task {name!r} is paired, but {', '.join(unpaired)} "
                    "was trained on single chains only"
                )
        if "dataset_columns" in fields:
            fields["dataset_columns"] = DatasetColumns(**fields["dataset_columns"])
        if "training_args" in fields:
            fields["training_args"] = TrainingArguments(**fields["training_args"])
        tasks.append(TASKS[kind](**fields))

    return RunSpec(output_dir=Path(raw["output_dir"]), models=models, tasks=tasks)
