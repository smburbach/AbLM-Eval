import yaml
import json
import pathlib
import subprocess
from importlib.resources import files
from dataclasses import asdict

from ..configs import ClassificationConfig
from ..tasks import classification

__all__ = ["create_results_dir"]


def _check_dir(path: pathlib.Path):
    """
    Raise exception if the given directory or any of its subdirectories contain files.
    """
    if path.exists() and any(p.is_file() for p in path.rglob("*")):
        raise Exception(f"The directory '{path}' exists and is not empty!")


def _user_prompt(prompt: str) -> bool:
    while True:
        response = input(prompt + " (yes/no): ").strip().lower()
        if response in ("yes", "no"):
            return response == "yes"
        print("Invalid response. Please enter 'yes' or 'no'.")


def _override_accelerate_config(task_name: str, task_dir: pathlib.Path):
    # get default
    default_config_path = files(classification).joinpath(
        "default_accelerate_config.yaml"
    )
    default_config = yaml.safe_load(default_config_path.read_text())

    # print default
    print("\nThe default accelerate config for classification is:")
    print(yaml.dump(default_config, sort_keys=False, default_flow_style=False))

    # prompt for custom config
    config_path = task_dir / f"{task_name}_accelerate_config.yaml"
    if _user_prompt(
        f"Would you like to setup a different accelerate config for {task_name} classification?"
    ):
        subprocess.run(["accelerate", "config", "--config_file", str(config_path)])
    else:
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, sort_keys=False)


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
        if isinstance(config, ClassificationConfig):
            _override_accelerate_config(config.dataset_name, task_path)
