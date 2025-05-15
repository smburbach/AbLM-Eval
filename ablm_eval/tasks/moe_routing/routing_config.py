from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["RoutingConfig"]


@dataclass
class RoutingConfig:
    """Task: Routing Analysis

    Perform routing analysis tasks on a provided dataset.

    Parameters
    ----------
    data_path : str
        Path to a .parquet or .csv file containing sequences.
    sequence_column : str, optional
        Column name containing full sequence (ie. no separator will be added).
        Either `sequence_column` or both `heavy_column` and `light_column` must be provided.
    heavy_column : str, optional
        Column name containing heavy chain sequences.
        Either `sequence_column` or both `heavy_column` and `light_column` must be provided.
    light_column : str, optional
        Column name containing light chain sequences.
        Either `sequence_column` or both `heavy_column` and `light_column` must be provided.
    heavy_cdr_column : str, default="cdr_mask_heavy"
        Column name containing heavy chain CDR masks.
    light_cdr_column : str, default="cdr_mask_light"
        Column name containing light chain CDR masks.
    id_column : str, default="sequence_id"
        Column name containing sequence IDs.
    keep_columns : list, default=[]
        List of columns to retain in the processed dataset. The `id_column`, `sequence_column`, `heavy_cdr_column`, and `light_cdr_column` will be appended automatically.
    separator : str, default=`<cls>`
        Separator token for paired sequences.
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
    return_sequence : bool, default=False
        Whether to return the processed sequence in the output.
    output_dir : str, optional
        Directory where routing analysis results will be saved.

    Attributes
    ----------
    config_type : str, default="routing_analysis"
        The type of configuration.
    task_dir : str
        The directory name for the task.
    name : str
        The human-readable name of the task.
    runner : Callable
        The function to run the routing analysis task.
    """

    config_type: str = field(init=False, default="routing_analysis")

    @property
    def task_dir(self):
        return "routing_analysis"

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self):
        from .routing_run import run_routing_analysis
        return run_routing_analysis

    # required
    data_path: str

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
