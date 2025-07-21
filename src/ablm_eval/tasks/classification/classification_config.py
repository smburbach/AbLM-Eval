import pathlib
import yaml
import subprocess
from importlib.resources import files
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Literal

from transformers import TrainingArguments

from ...utils import BaseDatasetConfig

__all__ = ["ClassificationConfig"]


@dataclass
class ClassificationConfig(BaseDatasetConfig):
    """Task: Classification

    Perform classification tasks on a provided dataset, using the Hugging Face Trainer.

    Parameters
    ----------
    data_path : Dict[int, Dict[str, str]]
        Dictionary containing the paths to the train and test datasets, organized by fold.
        Example: {0: {"train": "/path/to/train_fold0.csv", "test": "/path/to/test_fold0.csv"}}

    launcher : {"accelerate", "python"}, default="accelerate"
        Launcher to use to train classification models.
    num_folds : int, default=5
        Number of folds for cross-validation.
    num_classes : int, default=2
        Number of classes for classification.
    multi_class_average : {"macro", "micro"}, default="macro"
        Averaging method for multi-class classification metrics. Used only when `num_classes > 2`.
    positive_label : int, default=1
        Label considered as positive for binary classification.
    attention_classifier : bool, default=True
        Whether to use an attention-based classifier.
    manually_freeze_base : bool, default=False
        Whether to manually freeze the base model. Default behavior is handled by BALM.
    padding : bool or str, default="max_length"
        Padding strategy for tokenization.
    max_len : int, default=256
        Maximum sequence length for tokenization.
    truncate : bool, default=True
        Whether to truncate sequences longer than `max_len`.
    add_special_tokens : bool, default=True
        Whether to add special tokens during tokenization.
    num_proc : int, default=128
        Number of processes to use for data preprocessing.

    training_args : TrainingArguments, optional
        HuggingFace TrainingArguments object, for training classification models.
    wandb_project : str, optional
        Name of the Weights & Biases project.
    wandb_run_group : str, optional
        Name of the Weights & Biases run group.
    wandb_job_type : str, optional
        Type of job for Weights & Biases logging.
    run_name : str, optional
        Name of the training run.

    """

    config_type: str = field(init=False, default="classification")

    @property
    def task_dir(self):
        return f"{self.dataset_name}_classification"

    @property
    def runner(self):
        from .classification_run import run_classification

        return run_classification

    # required
    # takes a dict instead of a str
    data_path: Dict[int, Dict[str, str]]  # fold -> {"train": path, "test": path}

    # classification details
    launcher: Literal["accelerate", "python"] = "accelerate"
    num_folds: int = 5
    num_classes: int = 2
    multi_class_average: Literal["macro", "micro"] = (
        "macro"  # only used in num_classes > 2
    )
    positive_label: int = 1
    attention_classifier: bool = True  # extra model arg
    manually_freeze_base: bool = False  # balm handles by default

    # tokenization
    padding: Union[bool, str] = "max_length"
    max_len: int = 256
    truncate: bool = True
    add_special_tokens: bool = True
    num_proc: int = 128
    keep_columns: list = field(default_factory=lambda: ["label"])

    # training args
    run_name: Optional[str] = None
    training_args: Optional[TrainingArguments] = None

    # wandb
    # if report_to = "wandb" in training args
    wandb_project: str = None
    wandb_run_group: str = None
    wandb_job_type: str = None

    @staticmethod
    def override_accelerate_config(task_name: str, task_dir: pathlib.Path):
        from .. import classification

        # get default
        default_config_path = files(classification).joinpath(
            "default_accelerate_config.yaml"
        )
        default_config = yaml.safe_load(default_config_path.read_text())

        print("\nThe default accelerate config for classification is:")
        print(yaml.dump(default_config, sort_keys=False, default_flow_style=False))

        config_path = task_dir / f"{task_name}_accelerate_config.yaml"

        # validate user response
        while True:
            response = (
                input(
                    f"Would you like to setup a different accelerate config for {task_name} classification? (yes/no): "
                )
                .strip()
                .lower()
            )
            if response in ("yes", "no"):
                break
            print("Invalid response. Please enter 'yes' or 'no'.")

        if response == "yes":
            subprocess.run(
                ["accelerate", "config", "--config_file", str(config_path)],
                check=True,
            )
        else:
            with open(config_path, "w") as f:
                yaml.dump(default_config, f, sort_keys=False)
