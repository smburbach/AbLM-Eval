import pathlib

from ..configs import ClassificationConfig

__all__ = ["create_results_dir"]


def _check_dir(path: pathlib.Path):
    """
    Raise exception if the given directory or any of its subdirectories contain files.
    """
    if path.exists() and any(p.is_file() for p in path.rglob("*")):
        raise Exception(f"The directory '{path}' exists and is not empty!")


def create_results_dir(output_dir: str, configs: list, ignore_existing: bool):
    """
    Create directory structure for results.
    """
    # base directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # task directories
    for config in configs:
        # task dir path
        if type(config) == ClassificationConfig:
            task_path = output_path / f"{config.classification_name}_{config.task_dir}"
        else:
            task_path = output_path / f"{config.task_dir}"
        config.output_dir = task_path

        # make task dir
        if not ignore_existing:
            _check_dir(task_path)
        task_path.mkdir(exist_ok=True)

        # results dir inside task dir
        subdir_path = task_path / "results"
        subdir_path.mkdir(exist_ok=True)
