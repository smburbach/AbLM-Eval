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
    InferenceConfig: "Inference",
    PerPositionConfig: "Per-position Inference",
    ClassificationConfig: "Classification",
    MutationPredConfig: "Mutation Analysis",
    RoutingConfig: "Routing Analysis",
}

CONFIG_TASK_MAPPING = {
    InferenceConfig: run_inference,
    PerPositionConfig: run_per_pos,
    ClassificationConfig: run_classification,
    MutationPredConfig: run_mutation_analysis,
    RoutingConfig: run_routing_analysis,
}

UNDERLINE = "\033[4m"
RESET = "\033[0m"

def run_eval(configs: list):

    for itr, config in enumerate(configs, 1):
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
        print(f"\n{UNDERLINE}Running Task {itr}: {task_str}{RESET}")
        task_fn(config)
