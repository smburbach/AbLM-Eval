from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["NaturalnessConfig"]


@dataclass
class NaturalnessConfig:
    """Task: Naturalness Prediction

    Perform naturalness prediction tasks on a provided dataset.

    Parameters
    ----------
    data_path : str
        Path to a .parquet or .csv file containing sequences.
    dataset_name : str, optional
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
    max_len : int, optional
        Maximum sequence length for tokenization.
    truncate : bool, default=False
        Whether to truncate sequences longer than `max_len`.
    add_special_tokens : bool, default=False
        Whether to add special tokens during tokenization.
    num_proc : int, default=128
        Number of processes to use for data preprocessing.
    keep_columns : list, default=["sequence_id"]
        List of columns to retain in the processed dataset.
    output_dir : str, optional
        Directory where naturalness prediction results will be saved.

    Attributes
    ----------
    config_type : str, default="naturalness"
        The type of configuration.
    task_dir : str
        The directory name for the task.
    name : str
        The human-readable name of the task.
    runner : Callable
        The function to run the naturalness prediction task.
    """

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
