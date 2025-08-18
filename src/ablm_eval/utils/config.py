from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Literal

__all__ = ["DatasetColumns", "BaseDatasetConfig"]


@dataclass
class DatasetColumns:
    """
    Define custom column names for provided dataset.
    Values not provided with be set to defaults based on datatype and AIRR standards.

    Parameters
    ----------
    chain_names : list[str], optional
        Names to label each antibody chain (e.g., ["heavy", "light"] for paired data,
        ["unpaired"] for unpaired).
    chain_columns : list[str], optional
        Column names containing the amino acid sequences for each chain.
    cdr_columns : list[str], optional
        Column names indicating CDR masks for each chain.
    mutation_columns : list[str], optional
        Column names tracking the mutation counts for each chain.
    id_column : str, default="sequence_id"
        Identifier column name.
    locus_column: str, optional
        Column name containing the locus information for unpaired sequences.
    Methods
    -------
    apply_defaults(datatype)
        Fills unset fields with defaults based on antibody datatype.
    to_dict() -> dict[str, list[str]]
        Returns a dict of all non-None fields.
    """

    chain_names: Optional[List[str]] = None
    chain_columns: Optional[List[str]] = None
    cdr_columns: Optional[List[str]] = None
    mutation_columns: Optional[List[str]] = None
    id_column: Optional[str] = "sequence_id"
    locus_column: Optional[str] = None

    def apply_defaults(self, datatype: Literal["paired", "unpaired"]):
        # default values
        defaults = {
            "paired": {
                "chain_names": ["heavy", "light"],
                "chain_columns": ["sequence_aa_heavy", "sequence_aa_light"],
                "cdr_columns": ["cdr_mask_heavy", "cdr_mask_light"],
                "mutation_columns": [
                    "v_mutation_count_aa_heavy",
                    "v_mutation_count_aa_light",
                ],
            },
            "unpaired": {
                "chain_names": ["unpaired"],
                "chain_columns": ["sequence_aa"],
                "cdr_columns": ["cdr_mask"],
                "mutation_columns": ["v_mutation_count_aa"],
                "locus_column": "locus",
            },
        }

        # apply defaults only if the current value is None
        for field, default_value in defaults[datatype].items():
            current_value = getattr(self, field)
            if current_value is None:
                setattr(self, field, default_value)

    def to_dict(self) -> Dict[str, List[str]]:
        raw = asdict(self)
        return {k: v for k, v in raw.items() if v is not None}


@dataclass
class BaseDatasetConfig(ABC):
    """
    Configuration for loading and processing an antibody dataset.

    Subclasses must implement the `task_dir` property to specify
    the directory name used for model/task outputs.

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
        Short name for the dataset, used in naming output directories.
    separator : str, default '<cls>'
        Special token used to separate chain sequences.
    output_dir : str, optional
        Base directory for saving outputs.
    keep_columns : list[str], optional
        Additional columns to retain from the input. Required columns
        (ID, sequence, CDR/mutation) are added automatically.

    Raises
    ------
    ValueError
        If `antibody_datatype` is not 'paired' or 'unpaired'.
    """

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    @abstractmethod
    def task_dir(self) -> str:
        pass

    # data
    data_path: str
    antibody_datatype: Literal["paired", "unpaired"]
    dataset_columns: Optional[DatasetColumns] = None
    dataset_name: Optional[str] = None
    separator: str = "<cls>"
    tokenizer_path: str | None = None

    # output
    output_dir: Optional[str] = None
    keep_columns: List[str] = field(
        default_factory=list
    )  # id, sequence, cdr, and mutation columns will be appended

    def __post_init__(self):

        # validate antibody datatype
        if self.antibody_datatype not in ("paired", "unpaired"):
            raise ValueError(f"Invalid antibody_datatype: {self.antibody_datatype}")

        # apply column defaults if needed
        if self.dataset_columns is None:
            self.dataset_columns = DatasetColumns()
        self.dataset_columns.apply_defaults(self.antibody_datatype)

        # extend keep_columns based on the column overrides
        extra_cols = [
            col
            for col in (
                "sequence",
                self.dataset_columns.id_column,
                self.dataset_columns.locus_column,
                *(self.dataset_columns.chain_columns or []),
                *(self.dataset_columns.cdr_columns or []),
                *(self.dataset_columns.mutation_columns or []),
            )
            if col is not None
        ]
        self.keep_columns.extend(extra_cols)
