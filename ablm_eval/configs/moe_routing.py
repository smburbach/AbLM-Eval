from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["RoutingConfig"]


@dataclass
class RoutingConfig:
    config_type: str = field(init=False, default="routing_analysis")

    @property
    def task_dir(self):
        return "routing_analysis"

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self):
        from ..tasks import run_routing_analysis
        return run_routing_analysis

    @property
    def comparer(self):
        pass

    # required
    routing_data: str

    # data processing
    sequence_column: Optional[str] = None
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
