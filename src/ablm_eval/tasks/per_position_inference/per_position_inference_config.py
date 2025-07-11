from dataclasses import dataclass, field
from typing import Callable, Union

from ...utils import BaseDatasetConfig

__all__ = ["PerPositionConfig"]


@dataclass
class PerPositionConfig(BaseDatasetConfig):
    """Task: Per-position Inference

    Perform per-position inference, where each position in a sequence is iteratively masked.

    Parameters
    ----------
    data_path : str
        Path to the dataset file (CSV or parquet).
    antibody_datatype : {'paired', 'unpaired'}
        Format of antibody data, either paired or unpaired.
    dataset_columns : DatasetColumns, optional
        Defines column names for sequences, CDRs, mutations, etc.
        If None, defaults are generated based on `antibody_datatype`.
    dataset_name : str, optional
        Short name used for constructing `task_dir`.
    separator : str, default '<cls>'
        Special token used to separate chain sequences.
    output_dir : str, optional
        Base directory for saving outputs.

    padding : bool or str, default=False
        Controls padding behavior.
    max_len : int, optional
        Maximum tokenization length for input sequences.
    truncate : bool, default=False
        Whether to truncate sequences longer than `max_len`.
    add_special_tokens : bool, default=False
        Whether to include special tokens in tokenization.
    num_proc : int, default=128
        Number of parallel processes to use for dataset preparation.
    """

    config_type: str = field(init=False, default="per_pos_inference")

    @property
    def task_dir(self):
        return f"{self.dataset_name}_per_pos_inference"

    @property
    def runner(self) -> Callable:
        from .per_position_inference_run import run_per_pos

        return run_per_pos

    # tokenization
    padding: Union[bool, str] = False
    max_len: int = None
    truncate: bool = False
    add_special_tokens: bool = False
    num_proc: int = 128

    # output
    keep_columns: list = field(
        default_factory=lambda: ["locus"]
    )  # id, chain, cdr, and mutation columns will be appended
