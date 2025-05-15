from dataclasses import dataclass, field
from typing import Callable, Optional, Union

__all__ = ["InferenceConfig"]


@dataclass
class InferenceConfig:
    """Task: Inference

    Perform MLM inference on a provided dataset, using the Hugging Face Trainer.

    Parameters
    ----------
    data_path : str
        Path to a .parquet or .csv file containing sequences.
    dataset_name : str, default="test"
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
    keep_columns : list, default=[]
        List of columns to retain in the processed dataset. Default is an empty list.
    mlm : bool, default=True
        Whether to use masked language modeling (MLM).
    mlm_probability : float, optional
        Probability of masking tokens for MLM. Default is 0.15.
    batch_size : int, default=32
        Batch size for inference.
    return_moe_losses : bool, default=False
        Whether to return MoE (Mixture of Experts) losses for BALM-MoE models.
    report_to : str, default="none"
        Reporting destination for logging (e.g., "none", "wandb").
    output_dir : str, optional
        Directory where inference results will be saved. Default is None.

    Attributes
    ----------
    config_type : str, default="inference"
        The type of configuration.
    task_dir : str
        The directory name for the task.
    name : str
        The human-readable name of the task.
    runner : Callable
        The function to run the inference task.
    """

    config_type: str = field(init=False, default="inference")

    @property
    def task_dir(self) -> str:
        return "inference"

    @property
    def name(self) -> str:
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self) -> Callable:
        from .inference_run import run_inference
        return run_inference

    # required
    data_path: str
    dataset_name: str = "test"

    # data processing
    sequence_column: Optional[str] = None
    heavy_column: Optional[str] = None
    light_column: Optional[str] = None
    separator: str = "<cls>"

    # tokenization
    padding: Union[bool, str] = "max_length"
    max_len: int = 256
    truncate: bool = True
    add_special_tokens: bool = True
    num_proc: int = 128
    keep_columns: list = field(default_factory=list)

    # collator
    mlm: bool = True
    mlm_probability: float = 0.15

    # inference
    batch_size: int = 32
    return_moe_losses: bool = False
    report_to: str = "none"

    # output
    output_dir: str = None
