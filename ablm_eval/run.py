from pathlib import Path

import gc
import torch

from .utils import create_results_dir, config_from_json


__all__ = ["evaluate_ablms", "compare_results"]

BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"


def _clean_up():
    gc.collect()
    torch.cuda.empty_cache()


def _eval_model(model_name: str, model_path: str, configs: list):
    """
    Evaluates single model based on task configs.
    Should not be called by user directly, because it assumes the
    output directories have been created.
    """

    for itr, config in enumerate(configs, 1):
        print(f"{UNDERLINE}Running Task #{itr}: {config.name}{RESET}")
        task_fn = config.runner
        task_fn(model_name, model_path, config)
        _clean_up()


def compare_results(output_dir: str = None, configs: list = None):
    """
    Compare model results in a given output directory.
    """

    if output_dir is None and configs is None:
        raise ValueError("Either the output_dir or configs list must be provided.")

    # load configs from output_dir if not provided
    if configs is None:
        configs = []
        output_path = Path(output_dir)
        for folder_path in output_path.iterdir():
            if folder_path.is_dir() and not folder_path.name.startswith("."):
                config_file_path = folder_path / "config.json"
                config = config_from_json(str(config_file_path))
                configs.append(config)

    # evaluate
    for config in configs:
        compare_fn = config.comparer
        if compare_fn is None:
            continue
        compare_fn(config)


def evaluate_ablms(
    models: dict,
    configs: list,
    shared_output_dir: str = "./results",
    generate_comparisons: bool = True,
    ignore_existing_files: bool = False,
):
    """
    Evaluate (and optionally compare) the provided models on the
    given task configs.
    """

    # create output dirs
    create_results_dir(shared_output_dir, configs, ignore_existing_files)

    # eval
    for itr, (model_name, model_path) in enumerate(models.items(), 1):
        print(f"\n{BOLD}Evaluating Model #{itr}: {model_name}{RESET}")
        _eval_model(model_name, model_path, configs)

    # plot comparisons
    if generate_comparisons:
        print(f"\n{BOLD}Generating model comparisons...{RESET}")
        compare_results(configs=configs)
