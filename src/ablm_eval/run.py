import gc
from typing import Optional
from pathlib import Path

import torch

from .utils.directories import create_results_dir
from .tasks.compare_registry import (
    _config_from_json,
    _comparer_from_str,
)


__all__ = ["evaluate_ablms", "compare_results", "compare_task"]

BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"


def compare_task(
    task_type: str, task_results_dir: str, output_dir: Optional[str] = None
):
    """
    Compare the results of a single task, in the provided output directory.
    """
    # get compare fn
    compare_fn = _comparer_from_str(task_type)
    if compare_fn is None:
        return

    # if output dir not provided
    output_dir = output_dir or task_results_dir

    # run comparison fn
    compare_fn(results_dir=task_results_dir, output_dir=output_dir, task_str=task_type)


def compare_results(output_dir: Optional[str] = None, configs: Optional[list] = None):
    """
    Compare model results in a given output directory, assuming this directory
    follows the directory structure created in `evaluate_ablms`.
    """

    if bool(output_dir) == bool(configs):
        raise ValueError("Provide either `output_dir` or `configs`, but not both.")

    if output_dir:
        for folder in Path(output_dir).iterdir():
            if folder.is_dir() and not folder.name.startswith("."):
                config_path = folder / "config.json"
                if config_path.exists():
                    # extract info from config
                    config = _config_from_json(str(config_path))

                    # run comparison
                    compare_task(
                        task_type=config["task_type"],
                        task_results_dir=config["results_dir"],
                        output_dir=config["output_dir"],
                    )

    if configs:
        for config in configs:
            output_dir = config.output_dir
            compare_task(
                task_type=config.config_type,
                task_results_dir=f"{output_dir}/results/",
                output_dir=output_dir,
            )


def _clean_up():
    gc.collect()
    torch.cuda.empty_cache()


def _eval_model(model_name: str, model_path: str, configs: list):
    for itr, config in enumerate(configs, 1):
        # run task
        print(f"{UNDERLINE}Running Task #{itr}: {config.name}{RESET}")
        task_fn = config.runner
        task_fn(model_name, model_path, config)

        # clean up memory
        _clean_up()


def evaluate_ablms(
    models: dict,
    configs: list,
    shared_output_dir: str = "./results",
    generate_comparisons: bool = True,
    ignore_existing_files: bool = False,
):
    """
    Evaluate (and optionally compare) the provided models on the
    given tasks.
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
