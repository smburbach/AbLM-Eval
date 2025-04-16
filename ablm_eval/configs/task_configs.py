from dataclasses import dataclass
from typing import Optional

__all__ = [
    "EvalConfig",
    "InferenceConfig",
    "PerPositionConfig",
    "ClassificationConfig"
]


@dataclass
class EvalConfig:
    # model
    model_name: str
    model_path: str

    # dataset
    heavy_colomn: Optional[str] = None
    light_colomn: Optional[str] = None
    separator: str = "<cls>"

    # tokenization
    padding: Union[bool, str] = "max_length"
    max_len: int = 256
    truncate: bool = True
    add_special_tokens: bool = True
    num_proc: int = 128
    return_sequence: bool = False

    # output
    output_dir: str = "./results"


@dataclass
class InferenceConfig(EvalConfig):
    # dataset - required
    inference_data: str
    
    # collator
    mlm: bool = True
    mlm_probability: float = 0.15

    # balm moe
    return_moe_losses: bool = False

    # inference
    batch_size: int = 32
    report_to: str = "none"


@dataclass
class PerPositionConfig(EvalConfig):
    # dataset - required
    per_pos_data: str

    # tokenization (override base)
    padding = False
    truncate = False
    add_special_tokens = False
    return_sequence = True


@dataclass
class ClassificationConfig(EvalConfig):
    # dataset & classification details - required
    classification_name: str
    dataset_dir: str
    num_folds: int = 5
    num_classes: int = 2

    # extra model args
    attention_classifier: bool = True

    # training args
    label_column: str = "label"
    max_length: int = 512
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-4
    loss_fn: str = "cross_entropy"
    use_weighted_loss: bool = False
    metrics: Optional[list[str]] = None
