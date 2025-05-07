import json
from typing import Dict, Type

from ..plots import *

CONFIG_COMPARER_REGISTRY: Dict[str, Type] = {
    "classification": table_compare,
    "inference": table_compare,
    "per_pos_inference": per_pos_compare,
    "mutation_prediction": mut_analysis_compare,
    "routing_analysis": routing_compare,
    "naturalness": naturalness_compare,
}


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
