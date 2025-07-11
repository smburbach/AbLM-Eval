from dataclasses import dataclass, field
from typing import Callable, Union

from ...utils import BaseDatasetConfig

__all__ = ["InferenceConfig"]


@dataclass
class InferenceConfig(BaseDatasetConfig):
    """Task: Inference

    Perform MLM inference on a provided dataset, using the Hugging Face Trainer.

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

    padding : bool or str, default='max_length'
        Controls padding behavior.
    max_len : int, default=256
        Maximum tokenization length for input sequences.
    truncate : bool, default=True
        Whether to truncate sequences longer than `max_len`.
    add_special_tokens : bool, default=True
        Whether to include special tokens in tokenization.
    num_proc : int, default=128
        Number of parallel processes for dataset preprocessing.

    mlm : bool, default=True
        Whether to apply MLM masking during inference.
    mlm_probability : float, default=0.15
        Fraction of tokens to mask when using MLM.

    batch_size : int, default=32
        Batch size for the Trainer during inference.
    return_moe_losses : bool, default=False
        When using BALM MoE models, whether to return loss values per expert.
    report_to : str, default='none'
        Where to send logging/metrics during Trainer execution, ex. 'none', 'wandb'.
    """

    config_type: str = field(init=False, default="inference")

    @property
    def task_dir(self) -> str:
        return "inference"

    @property
    def runner(self) -> Callable:
        from .inference_run import run_inference

        return run_inference

    # tokenization
    padding: Union[bool, str] = "max_length"
    max_len: int = 256
    truncate: bool = True
    add_special_tokens: bool = True
    num_proc: int = 128

    # collator
    mlm: bool = True
    mlm_probability: float = 0.15

    # inference
    batch_size: int = 32
    return_moe_losses: bool = False
    report_to: str = "none"
