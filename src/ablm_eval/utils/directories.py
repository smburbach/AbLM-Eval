import yaml
import json
import pathlib
import subprocess
from importlib.resources import files
from dataclasses import asdict

__all__ = ["create_results_dir"]


def _check_dir(path: pathlib.Path):
    """
    Raise exception if the given directory or any of its subdirectories contain files.
    """
    if path.exists() and any(p.is_file() for p in path.rglob("*")):
        raise Exception(f"The directory '{path}' exists and is not empty!")


def create_results_dir(output_dir: str, configs: list, ignore_existing: bool):
    """
    Create directory structure for results.
    """
    # base directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # task directories
    for config in configs:
        # task dir path
        task_path = output_path / f"{config.task_dir}"
        config.output_dir = str(task_path)

        # make task dir
        if not ignore_existing:
            _check_dir(task_path)
        task_path.mkdir(exist_ok=True)

        # results dir inside task dir
        subdir_path = task_path / "results"
        subdir_path.mkdir(exist_ok=True)

        # save config
        with open(f"{task_path}/config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

        # add accelerate config to classification task dir
        from ..tasks import ClassificationConfig

        if (
            isinstance(config, ClassificationConfig)
            and getattr(config, "launcher") == "accelerate"
        ):
            ClassificationConfig.override_accelerate_config(
                config.dataset_name, task_path
            )
