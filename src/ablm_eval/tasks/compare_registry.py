import json
from typing import Dict, Type

CONFIG_COMPARER_REGISTRY: Dict[str, Type] = {}


def register_comparer(task_name: str, comparer_func: Type):
    """
    Registers a comparer function for a specific task.
    """
    CONFIG_COMPARER_REGISTRY[task_name] = comparer_func


def _comparer_from_str(task_str: str):
    """
    Loads the eval comparer based on task string.
    """
    if task_str not in CONFIG_COMPARER_REGISTRY.keys():
        valid_tasks = ", ".join(CONFIG_COMPARER_REGISTRY.keys())
        raise ValueError(f"The task string must be one of the following: {valid_tasks}")
    return CONFIG_COMPARER_REGISTRY[task_str]


def _config_from_json(path: str):
    with open(path, "r") as f:
        config = json.load(f)

    output_dir = config["output_dir"]
    return {
        "task_type": config["config_type"],
        "output_dir": output_dir,
        "results_dir": f"{output_dir}/results/",
    }
