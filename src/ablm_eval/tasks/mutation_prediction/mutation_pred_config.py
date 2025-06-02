from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["MutationPredConfig"]


@dataclass
class MutationPredConfig:
    """Task: Mutation Prediction

    Perform mutation prediction tasks on a provided dataset.

    Parameters
    ----------
    data_path : str
        Path to a .parquet or .csv file containing sequences.
    dataset_name : str, optional
        Name of the dataset being used.
    data_processed : bool, default=False
        Whether the dataset has already been processed.
            If False, assumes the data is formatted with one row for paired sequences,
            with AIRR columns names appended with '_heavy' and '_light'.
    sequence_column : str, default="sequence_germ"
        Sequence column to be masked in per-pos inference.
        Can be either "sequence_germ" or "sequence_mutated".
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
    keep_columns : list, default=[
        "sequence_id",
        "v_mutation_count_aa_heavy",
        "v_mutation_count_aa_light",
        "sequence_mutated",
        "sequence_germ",
    ]
        List of columns to retain in the processed dataset.
    output_dir : str, optional
        Directory where mutation prediction results will be saved.

    Attributes
    ----------
    config_type : str, default="mutation_prediction"
        The type of configuration.
    task_dir : str
        The directory name for the task.
    name : str
        The human-readable name of the task.
    runner : Callable
        The function to run the mutation prediction task.
    """

    config_type: str = field(init=False, default="mutation_prediction")

    @property
    def task_dir(self):
        return "mutation_prediction"

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self):
        from .mutation_pred_run import run_mutation_pred
        return run_mutation_pred

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
