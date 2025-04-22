from dataclasses import dataclass, field
from typing import Optional, Union


__all__ = [
    "InferenceConfig",
    "PerPositionConfig",
    "ClassificationConfig",
    "MutationPredConfig",
    "RoutingConfig",
]


def _name_from_task_dir(task_dir: str) -> str:
    return task_dir.replace("_", " ").title()


@dataclass
class InferenceConfig:
    @property
    def task_dir(self):
        return "inference"

    @property
    def name(self):
        return _name_from_task_dir(self.task_dir)

    @property
    def runner(self):
        from ..tasks import run_inference
        return run_inference

    # required
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
    keep_columns: list = field(default_factory=list)

    # collator
    mlm: bool = True
    mlm_probability: float = 0.15

    # inference
    batch_size: int = 32
    return_moe_losses: bool = False  # balm moe
    report_to: str = "none"

    # output
    output_dir: str = None


@dataclass
class PerPositionConfig:
    @property
    def task_dir(self):
        return "per_pos_inference"

    @property
    def name(self):
        return _name_from_task_dir(self.task_dir)

    @property
    def runner(self):
        from ..tasks import run_per_pos
        return run_per_pos

    # required
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
    keep_columns: list = field(default_factory=lambda: ["sequence"])

    # output
    output_dir: str = None


@dataclass
class ClassificationConfig:
    @property
    def task_dir(self):
        return "classification"

    @property
    def name(self):
        return _name_from_task_dir(self.task_dir)

    @property
    def runner(self):
        from ..tasks import run_classification
        return run_classification

    # required
    dataset_dir: str
    file_prefix: str  # ie. 'hd-0_cov-1' for the file 'hd-0_cov-1_train{i}.csv'
    classification_name: str

    # data processing
    heavy_column: Optional[str] = None
    light_column: Optional[str] = None
    separator: str = "<cls>"

    # classification details
    num_folds: int = 5
    num_gpus_per_fold: int = 1
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
    logging_first_step: bool = (True,)

    # output
    output_dir: str = None


@dataclass
class MutationPredConfig:
    @property
    def task_dir(self):
        return "mutation_prediction"

    @property
    def name(self):
        return _name_from_task_dir(self.task_dir)

    @property
    def runner(self):
        from ..tasks import run_mutation_analysis
        return run_mutation_analysis

    # required
    mutation_data: str

    # output
    output_dir: str = None


@dataclass
class RoutingConfig:
    @property
    def task_dir(self):
        return "routing_analysis"

    @property
    def name(self):
        return _name_from_task_dir(self.task_dir)

    @property
    def runner(self):
        from ..tasks import run_routing_analysis
        return run_routing_analysis

    # required
    routing_data: str

    # data processing
    heavy_column: Optional[str] = None
    light_column: Optional[str] = None
    heavy_cdr_column: Optional[str] = "cdr_mask_heavy"
    light_cdr_column: Optional[str] = "cdr_mask_light"
    id_column: Optional[str] = "sequence_id"
    keep_columns: list = field(
        default_factory=list
    )  # id, sequence, heavy_cdr, and light_cdr columns will be appended
    separator: str = "<cls>"

    # tokenization
    padding: Union[bool, str] = "max_length"
    max_len: int = 256
    truncate: bool = True
    add_special_tokens: bool = True
    num_proc: int = 128
    return_sequence: bool = False

    # output
    output_dir: str = None
