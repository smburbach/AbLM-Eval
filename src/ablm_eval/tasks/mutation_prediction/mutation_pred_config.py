from dataclasses import dataclass, field
from typing import Union, Literal

from ...utils import BaseDatasetConfig

__all__ = ["MutationPredConfig"]


@dataclass
class MutationPredConfig(BaseDatasetConfig):
    """Task: Mutation Prediction

    Perform mutation prediction tasks on a provided dataset.

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

    data_processed: bool, default=False
        Whether or not the data is processed (mutated and germline aligned).
    sequence_column: {'sequence_germ', 'sequence_mutated'}
        Whether to provide the germline or mutated sequence to the model during inference.
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

    config_type: str = field(init=False, default="mutation_prediction")

    @property
    def task_dir(self):
        return f"{self.dataset_name}_mutation_prediction"

    @property
    def runner(self):
        from .mutation_pred_run import run_mutation_pred

        return run_mutation_pred

    # data processing
    data_processed: bool = False
    sequence_column: Literal["sequence_germ", "sequence_mutated"] = "sequence_germ"

    # tokenization
    padding: Union[bool, str] = False
    max_len: int = None
    truncate: bool = False
    add_special_tokens: bool = True
    num_proc: int = 128
