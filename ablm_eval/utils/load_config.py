import json
from typing import Dict, Type

from ..configs import *

__all__ = ["config_from_json"]

CONFIG_TYPE_REGISTRY: Dict[str, Type] = {
    "classification": ClassificationConfig,
    "inference": InferenceConfig,
    "per_pos_inference": PerPositionConfig,
    "mutation_prediction": MutationPredConfig,
    "routing_analysis": RoutingConfig,
}


def config_from_json(path: str):
    """
    Loads the correct config type based on the type in the JSON file.
    """

    with open(path, "r") as f:
        data = json.load(f)

    # get config type
    config_type = data.pop("config_type")
    if config_type not in CONFIG_TYPE_REGISTRY:
        raise ValueError(f"Unknown config type: {config_type}")

    # get config class
    config_class = CONFIG_TYPE_REGISTRY[config_type]

    # load
    return config_class(**data)
