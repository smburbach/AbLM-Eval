from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["PerPositionConfig"]


@dataclass
class PerPositionConfig:
    config_type: str = field(init=False, default="per_pos_inference")

    @property
    def task_dir(self):
        return "per_pos_inference"

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self):
        from ..tasks import run_per_pos
        return run_per_pos

    @property
    def comparer(self):
        from ..plots import per_pos_compare
        return per_pos_compare

    # required
    per_pos_data: str

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
        default_factory = lambda: [
            "sequence_id",
            "sequence",
            "cdr_mask_heavy",
            "cdr_mask_light",
            "v_mutation_count_aa_heavy",
            "v_mutation_count_aa_light",
        ]
    )

    # output
    output_dir: str = None
