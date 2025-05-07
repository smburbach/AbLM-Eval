from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["NaturalnessConfig"]


@dataclass
class NaturalnessConfig:
    config_type: str = field(init=False, default="naturalness")

    @property
    def task_dir(self):
        return "naturalness"

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self):
        from .naturalness_run import run_naturalness
        return run_naturalness

    # required
    data_path: str
    dataset_name: str = None

    # data processing
    sequence_column: Optional[str] = None
    heavy_column: Optional[str] = None
    light_column: Optional[str] = None
    separator: str = "<cls>"

    # tokenization
    padding: Union[bool, str] = False
    max_len: int = None
    truncate: bool = False
    add_special_tokens: bool = False
    num_proc: int = 128
    keep_columns: list = field(
        default_factory=lambda: [
            "sequence_id",
        ]
    )

    # output
    output_dir: str = None
