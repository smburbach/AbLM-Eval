import argparse
from dataclasses import asdict
from typing import Union

from .configs import EvalConfig, InferenceConfig, PerPositionConfig
from ..tasks import run_inference, run_per_pos

__all__ = [
    "create_parser",
    "load_config"
]


TASK_CONFIG_MAPPING = {
    "inference": InferenceConfig,
    "per_pos_inference": PerPositionConfig,
}


# define parser based on configs
def create_parser():
    parser = argparse.ArgumentParser()

    # add all task config params as accepted arguments
    for config_class in [EvalConfig, InferenceConfig, PerPositionConfig]:
        for field in fields(config_class):
            parser.add_argument(f'--{field.name}', type=field.type)

    # task selector
    parser.add_argument(
        '--task', 
        type=str, 
        choices=["inference", "per_pos"], 
        required=True, 
        help="Task to perform"
    )

    return parser


def _override_config_with_args(config, args):
    for field in dataclasses.fields(config):
        val = getattr(args, field.name, None)
        if val is not None:
            setattr(config, field.name, val)
    return config


def load_config(args: argparse.Namespace) -> Union[InferenceConfig, PerPositionConfig]:
    """
    Load the appropriate config based on the task and override with command-line args.
    """
    # Retrieve the config class based on the task
    config_class = TASK_CONFIG_MAPPING.get(args.task)
    
    if config_class is None:
        raise ValueError(f"Unknown task: {args.task}")

    # Create the config instance based on the task
    config = config_class(
        model_name=args.model_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
    )

    # CLI override
    config = _override_config_with_args(config, args)
    
    return config
