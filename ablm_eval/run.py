import yaml
import subprocess
from importlib.resources import files

from .utils import single_model_dir, multiple_models_dir
from .configs import ClassificationConfig
from .tasks import classification

__all__ = ["eval_and_compare", "eval_model"]

BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"


def _user_prompt(prompt: str) -> bool:
    while True:
        response = input(prompt + " (yes/NO): ").strip().lower()
        if response in ("yes", "no"):
            return response == "yes"
        print("Invalid response. Please enter 'yes' or 'no'.")


def _override_accelerate_config(task_name: str, task_dir: str):
    # get default
    default_config_path = files(classification).joinpath(
        "default_accelerate_config.yaml"
    )
    default_config = yaml.safe_load(default_config_path.read_text())

    # print default
    print("\nThe default accelerate config for classification is:")
    print(yaml.dump(default_config, sort_keys=False, default_flow_style=False))

    # prompt for custom config
    config_path = f"{task_dir}/{task_name}_accelerate_config.yaml"
    if _user_prompt(
        f"Would you like to setup a different accelerate config for {task_name} classification?"
    ):
        subprocess.run(["accelerate", "config", "--config_file", config_path])
    else:
        with open(config_path, "w") as f:
            yaml.dump(default_config, f, sort_keys=False)


def eval_model(model_name: str, model_path: str, configs: list):
    for itr, config in enumerate(configs, 1):
        print(f"\n{UNDERLINE}Running Task #{itr}: {config.name}{RESET}")
        task_fn = config.runner
        task_fn(model_name, model_path, config)


def compare_models(configs: list, models: dict):
    print(f"\nGenerating model comparisons...")
    for config in configs:
        compare_fn = config.comparer
        compare_fn(config, models)


def eval_and_compare(
    models: dict,
    configs: list,
    shared_output_dir: str = "./results",
    ignore_existing_files=False,
):

    # create output directory
    if len(models) == 1:
        single_model_dir(shared_output_dir, ignore_existing_files)
    else:
        multiple_models_dir(shared_output_dir, configs, ignore_existing_files)

    for config in configs:
        config.output_dir = (
            str(config.output_dir)
            if config.output_dir is not None
            else str(shared_output_dir)
        )
        if isinstance(config, ClassificationConfig):
            _override_accelerate_config(config.classification_name, config.output_dir)

    for model_name, model_path in models:
        print(f"\n{BOLD}Evaluating Model: {model_name}{RESET}")
        eval_model(model_name, model_path, configs)

    if len(models) > 1:
        compare_models(configs, models)
