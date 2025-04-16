import sys
import argparse
import pathlib

from .configs import (
    create_parser,
    load_config
)
from .tasks import (
    run_inference,
    run_per_pos,
)

def main():
    # parser
    parser = create_parser()
    args = parser()

    # load config
    config = load_config(args)

    # build outputs dir: output_dir/model_name/
    output_path = pathlib.Path(config.output_dir) / config.model_name
    if output_path.exists() and any(output_path.iterdir()):
        if config.model_name != "test_model": # exception for testing
            raise Exception(
                f"The folder '{output_path}' already exists and is not empty! Please provide a different output directory or model name."
            )

    # tasks
    for task in config.task:
        if task == "inference":
            run_inference(config)
        elif task == "per_pos_inference":
            run_per_pos(config)
        elif task == "classification":
            run_classification(config)
        else:
            raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
