from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["MutationPredConfig"]


@dataclass
class MutationPredConfig:
    config_type: str = field(init=False, default="mutation_prediction")

    @property
    def task_dir(self):
        return "mutation_prediction"

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self):
        from ..tasks import run_mutation_analysis
        return run_mutation_analysis

    # required
    data_path: str
    dataset_name: str = None
    data_processed: bool = False

    # data processing
    sequence_column: Optional[str] = "sequence_germ"
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
            "v_mutation_count_aa_heavy",
            "v_mutation_count_aa_light",
            "sequence_mutated",
            "sequence_germ",
        ]
    )

    # output
    output_dir: str = None
