from .utils import single_model_dir, multiple_models_dir

__all__ = ["eval_and_compare", "eval_model"]

BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"


def eval_model(model_name: str, model_path: str, configs: list, shared_output_dir):
    for itr, config in enumerate(configs, 1):
        if config.output_dir is None:  # only 1 model
            config.output_dir = shared_output_dir

        # get task function
        task_fn = config.runner

        # run
        print(f"\n{UNDERLINE}Running Task #{itr}: {config.name}{RESET}")
        task_fn(model_name, model_path, config)


def eval_and_compare(models: dict, configs: list, shared_output_dir: str = "./results"):

    # create output directory
    if len(models) == 1:  # all
        single_model_dir(shared_output_dir)
    else:
        multiple_models_dir(shared_output_dir, configs)

    for model_name, model_path in models:
        print(f"\n{BOLD}Evaluating Model: {model_name}{RESET}")
        eval_model(model_name, model_path, configs, shared_output_dir)

    # TODO: plots / tables comparing models
