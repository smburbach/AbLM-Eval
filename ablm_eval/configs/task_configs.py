from dataclasses import dataclass
from typing import Optional, Union

__all__ = [
    "InferenceConfig",
    "PerPositionConfig",
    "ClassificationConfig",
    "MutationPredConfig",
    "RoutingConfig",
]


@dataclass
class InferenceConfig:
    # required
    model_name: str
    model_path: str
    inference_data: str

    # data processing
    heavy_column: Optional[str] = None
    light_column: Optional[str] = None
    separator: str = "<cls>"

    # tokenization
    padding: Union[bool, str] = "max_length"
    max_len: int = 256
    truncate: bool = True
    add_special_tokens: bool = True
    num_proc: int = 128
    return_sequence: bool = False

    # collator
    mlm: bool = True
    mlm_probability: float = 0.15

    # inference
    batch_size: int = 32
    return_moe_losses: bool = False  # balm moe
    report_to: str = "none"

    # output
    output_dir: str = "./results"


@dataclass
class PerPositionConfig:
    # required
    model_name: str
    model_path: str
    per_pos_data: str

    # data processing
    heavy_column: Optional[str] = None
    light_column: Optional[str] = None
    separator: str = "<cls>"

    # tokenization
    padding: Union[bool, str] = False
    max_len: int = None
    truncate: bool = False
    add_special_tokens: bool = False
    num_proc: int = 128
    return_sequence: bool = True

    # output
    output_dir: str = "./results"


@dataclass
class ClassificationConfig:
    # required
    model_name: str
    model_path: str
    dataset_dir: str
    file_prefix: str # ie. 'hd-0_cov-1' for the file 'hd-0_cov-1_train{i}.csv'
    classification_name: str

    # data processing
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
    return_sequence: bool = False

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
    logging_first_step: bool = True,

    # output
    output_dir: str = "./results"


@dataclass
class MutationPredConfig:
    # required
    model_name: str
    model_path: str
    mutation_data: str

    # output
    output_dir: str = "./results"


@dataclass
class RoutingConfig:
    # required
    model_name: str
    model_path: str
    routing_data: str

    # output
    output_dir: str = "./results"
