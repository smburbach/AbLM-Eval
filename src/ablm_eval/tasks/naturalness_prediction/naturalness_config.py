from dataclasses import dataclass, field
from typing import Union

from ...utils import BaseDatasetConfig

__all__ = ["NaturalnessConfig"]


@dataclass
class NaturalnessConfig(BaseDatasetConfig):
    """Task: Naturalness Prediction

    Perform naturalness prediction tasks on a provided dataset.

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
    separator : str, default='<cls>'
        Special token used to separate chain sequences.
    output_dir : str, optional
        Base directory for saving outputs.

    padding : bool or str, default="max_length"
        Controls padding behavior.
    max_len : int, default=256
        Maximum tokenization length for input sequences.
    truncate : bool, default=True
        Whether to truncate sequences longer than `max_len`.
    add_special_tokens : bool, default=True
        Whether to include special tokens in tokenization.
    num_proc : int, default=128
        Number of parallel processes to use for dataset preparation.
    """

    config_type: str = field(init=False, default="naturalness")

    @property
    def task_dir(self):
        return "naturalness"

    @property
    def runner(self):
        from .naturalness_run import run_naturalness

        return run_naturalness

    # tokenization
    padding: Union[bool, str] = False
    max_len: int = None
    truncate: bool = False
    add_special_tokens: bool = True
    num_proc: int = 128
