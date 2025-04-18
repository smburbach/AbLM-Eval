from .configs import (
    InferenceConfig,
    PerPositionConfig,
    ClassificationConfig,
    MutationPredConfig,
    RoutingConfig,
)
from .tasks import (
    run_inference,
    run_per_pos,
    run_classification,
    run_mutation_analysis,
    run_routing_analysis,
)
from .utils import create_dir

__all__ = ["run_eval"]


CONFIG_STR_MAPPING = {
    InferenceConfig: "inference",
    PerPositionConfig: "per-position-inference",
    ClassificationConfig: "classification",
    MutationPredConfig: "mutation-analysis",
    RoutingConfig: "routing-analysis",
}

CONFIG_TASK_MAPPING = {
    InferenceConfig: run_inference,
    PerPositionConfig: run_per_pos,
    ClassificationConfig: run_classification,
    MutationPredConfig: run_mutation_analysis,
    RoutingConfig: run_routing_analysis,
}


def run_eval(configs: list):

    for config in configs:
        # create output dir
        create_dir(config)

        # get task str
        task_str = CONFIG_STR_MAPPING.get(type(config))

        # get task function
        task_fn = CONFIG_TASK_MAPPING.get(type(config))
        if task_str is None or task_fn is None:
            raise ValueError(
                f"Cannot map the config type ('{type(config)}') to an eval function."
            )

        # run
        print(f"Running {task_str}")
        task_fn(config)
