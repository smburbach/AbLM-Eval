import gc
import torch

from .utils import create_results_dir


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


def compare_results(configs: list, models: dict):
    """
    Compare models the results in a given output directory.
    This can be called by the user.
    """
    for config in configs:
        compare_fn = config.comparer
        if compare_fn is None:
            continue
        compare_fn(config, models=models)


def evaluate_ablms(
    models: dict,
    configs: list,
    shared_output_dir: str = "./results",
    generate_comparisons: str = True,
    ignore_existing_files=False,
):

    # create output dirs
    create_results_dir(shared_output_dir, configs, ignore_existing_files)

    # eval
    for itr, (model_name, model_path) in enumerate(models, 1):
        print(f"\n{BOLD}Evaluating Model #{itr}: {model_name}{RESET}")
        _eval_model(model_name, model_path, configs)

    # plot comparisons
    if generate_comparisons:
        print(f"\n{BOLD}Generating model comparisons...{RESET}")
        compare_results(configs, models)
