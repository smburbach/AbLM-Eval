from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["InferenceConfig"]


@dataclass
class InferenceConfig:
    config_type: str = field(init=False, default="inference")

    @property
    def task_dir(self):
        return "inference"

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self):
        from ..tasks import run_inference
        return run_inference

    @property
    def comparer(self):
        from ..plots import table_compare
        return table_compare

    # required
    inference_data: str

    # data processing
    sequence_column: Optional[str] = None
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
