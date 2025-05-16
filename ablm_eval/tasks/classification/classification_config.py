from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["ClassificationConfig"]


@dataclass
class ClassificationConfig:
    """Task: Classification

    Perform classification tasks on a provided dataset, using the Hugging Face Trainer.

    Parameters
    ----------
    dataset_dir : str
        Path to the directory containing the dataset files.
    file_prefix : str
        Prefix for the dataset files (e.g., 'hd-0_cov-1' for the file 'hd-0_cov-1_train{i}.csv', where i is the fold).
    dataset_name : str
        Name of the dataset being used.
    sequence_column : str, optional
        Column name containing full sequence (ie. no separator will be added). 
        Either `sequence_column` or both `heavy_column` and `light_column` must be provided.
    heavy_column : str, optional
        Column name containing heavy chain sequences. 
        Either `sequence_column` or both `heavy_column` and `light_column` must be provided.
    light_column : str, optional
        Column name containing light chain sequences. 
        Either `sequence_column` or both `heavy_column` and `light_column` must be provided.
    separator : str, default=`<cls>`
        Separator token for paired sequences.
    num_folds : int, default=5
        Number of folds for cross-validation.
    num_classes : int, default=2
        Number of classes for classification.
    multi_class_average : str, default="macro"
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
    keep_columns : list, default=["label"]
        List of columns to retain in the processed dataset.
    report_to : str, default="wandb"
        Reporting destination for logging (e.g., "wandb").
    wandb_project : str, optional
        Name of the Weights & Biases project.
    wandb_run_group : str, optional
        Name of the Weights & Biases run group.
    wandb_job_type : str, optional
        Type of job for Weights & Biases logging.
    run_name : str, optional
        Name of the training run.
    seed : int, default=42
        Random seed for reproducibility.
    bf16 : bool, default=True
        Whether to use bfloat16 precision for training.
    fp16 : bool, default=False
        Whether to use float16 precision for training.
    learning_rate : float, default=1e-4
        Learning rate for training.
    train_batch_size : int, default=32
        Batch size for training.
    epochs : int, default=3
        Number of training epochs.
    warmup_ratio : float, default=0.1
        Warmup ratio for the learning rate scheduler.
    lr_scheduler_type : str, default="linear"
        Type of learning rate scheduler.
    eval_strategy : str, default="steps"
        Evaluation strategy during training.
    eval_steps : int, default=250
        Number of steps between evaluations.
    eval_batch_size : int, default=128
        Batch size for evaluation.
    eval_accumulation_steps : int, default=50
        Number of steps for gradient accumulation during evaluation.
    logging_steps : int, default=50
        Number of steps between logging events.
    save_strategy : str, default="no"
        Strategy for saving checkpoints.
    logging_first_step : bool, default=True
        Whether to log the first training step.
    output_dir : str, optional
        Directory where classification results will be saved.

    Attributes
    ----------
    config_type : str, default="classification"
        The type of configuration.
    task_dir : str
        The directory name for the task.
    name : str
        The human-readable name of the task.
    runner : Callable
        The function to run the classification task.
    """

    config_type: str = field(init=False, default="classification")

    @property
    def task_dir(self):
        return f"{self.dataset_name}_classification"

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self):
        from .classification_run import run_classification
        return run_classification

    # required
    dataset_dir: str
    file_prefix: str
    dataset_name: str

    # data processing
    sequence_column: Optional[str] = None
    heavy_column: Optional[str] = None
    light_column: Optional[str] = None
    separator: str = "<cls>"

    # classification details
    num_folds: int = 5
    num_classes: int = 2
    multi_class_average: str = "macro"  # only used in num_classes > 2
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

    # wandb
    report_to: str = "wandb"
    wandb_project: str = None
    wandb_run_group: str = None
    wandb_job_type: str = None

    # training args
    run_name: str = None
    seed: int = 42
    bf16: bool = True
    fp16: bool = False
    learning_rate: float = 1e-4
    train_batch_size: int = 32
    epochs: int = 3
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    eval_strategy: str = "steps"
    eval_steps: int = 250
    eval_batch_size: int = 128
    eval_accumulation_steps: int = 50
    logging_steps: int = 50
    save_strategy: str = "no"
    logging_first_step: bool = True

    # output
    output_dir: str = None
