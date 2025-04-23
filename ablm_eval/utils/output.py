import pathlib

from ..configs import ClassificationConfig

__all__ = ["single_model_dir", "multiple_models_dir"]


def _check_dir(path: pathlib.Path):
    """
    Raise exception if the given directory or any of its subdirectories contain files.
    """
    if path.exists() and any(p.is_file() for p in path.rglob("*")):
        raise Exception(f"The directory '{path}' exists and is not empty!")


def _results_dirs(base_dir: pathlib.Path):
    """
    Create results subdirectory.
    """
    subdir_path = base_dir / "results"
    subdir_path.mkdir(exist_ok=True)


def single_model_dir(output_dir: str, ignore_existing: bool):
    """
    Create directory structure for a single model.
    """
    output_path = pathlib.Path(output_dir)
    if not ignore_existing:
        _check_dir(output_path)

    # create dirs
    output_path.mkdir(exist_ok=True)
    _results_dirs(output_path)


def multiple_models_dir(output_dir: str, configs: list, ignore_existing: bool):
    """
    Create directory structure for multiple models.
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)
    for config in configs:
        if type(config) == ClassificationConfig:
            task_path = output_path / f"{config.classification_name}_{config.task_dir}"
        else:
            task_path = output_path / f"{config.task_dir}"
        config.output_dir = task_path

        # task dir
        if not ignore_existing:
            _check_dir(task_path)
        task_path.mkdir(exist_ok=True)

        # results dirs inside task dir
        _results_dirs(task_path)
