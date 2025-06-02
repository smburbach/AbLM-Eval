from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["PerPositionConfig"]


@dataclass
class PerPositionConfig:
    """Task: Per-position Inference

    Perform per-position inference, where each position in a sequence is iteratively masked, on a provided dataset.

    Parameters
    ----------
    data_path : str
        Path to a .parquet or .csv file containing sequences.
    dataset_name : str, default=None
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
    padding : bool or str, default=False
        Padding strategy for tokenization.
    max_len : int, default=None
        Maximum sequence length for tokenization.
    truncate : bool, default=False
        Whether to truncate sequences longer than `max_len`.
    add_special_tokens : bool, default=False
        Whether to add special tokens during tokenization.
    num_proc : int, default=128
        Number of processes to use for data preprocessing.
    keep_columns : list, default=[
        "sequence_id",
        "sequence",
        "cdr_mask_heavy",
        "cdr_mask_light",
        "v_mutation_count_aa_heavy",
        "v_mutation_count_aa_light",
    ]
        List of columns to retain in the processed dataset.
    output_dir : str, default=None
        Directory where per-position inference results will be saved.

    Attributes
    ----------
    config_type : str, default="per_pos_inference"
        The type of configuration.
    task_dir : str
        The directory name for the task.
    name : str
        The human-readable name of the task.
    runner : Callable
        The function to run the per-position inference task.
    """

    config_type: str = field(init=False, default="per_pos_inference")

    @property
    def task_dir(self):
        return "per_pos_inference"

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self):
        from .per_position_inference_run import run_per_pos

        return run_per_pos

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
            "sequence",
            "cdr_mask_heavy",
            "cdr_mask_light",
            "v_mutation_count_aa_heavy",
            "v_mutation_count_aa_light",
        ]
    )

    # output
    output_dir: str = None
