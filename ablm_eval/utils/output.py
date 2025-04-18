import pathlib

__all__ = ["single_model_dir", "multiple_models_dir"]


def _results_dirs(base_dir: pathlib.Path):
    # results
    results_path = base_dir / "results"
    results_path.mkdir()
    # plots
    plots_path = base_dir / "plots"
    plots_path.mkdir()


def _create_base_dir(base_dir: pathlib.Path):
    # check if directory exists and is not empty
    if base_dir.exists() and any(base_dir.iterdir()):
        raise Exception(f"The directory '{base_dir}' already exists and is not empty!")

    # create directory
    base_dir.mkdir(exist_ok=True)


def single_model_dir(output_dir: str):
    output_path = pathlib.Path(output_dir)
    _create_base_dir(output_path)
    _results_dirs(output_path)


def multiple_models_dir(output_dir: str, configs: list):
    output_path = pathlib.Path(output_dir)
    _create_base_dir(output_path)
    for config in configs:
        task_path = output_path / config.task_dir
        task_path.mkdir()
        _results_dirs(task_path)
        config.output_dir = task_path
