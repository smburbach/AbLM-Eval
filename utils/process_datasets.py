from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = ["load_and_tokenize"]


def _generate_sequence(dataset, split_name, heavy_column, light_column, separator):
    """
    Handle the logic of combining heavy_column and light_column into 'sequence' if necessary.
    """
    if heavy_column and light_column:
        if (
            heavy_column in dataset[split_name].column_names
            and light_column in dataset[split_name].column_names
        ):
            # concat heavy_column and light_column to create the 'sequence' column
            dataset = dataset.map(
                lambda x: {"sequence": x[heavy_column] + separator + x[light_column]},
                batched=True,
            )
        else:
            raise ValueError(
                f"Both columns {heavy_column} and {light_column} must exist in the dataset."
            )
    else:
        raise ValueError(
            "The dataset does not contain a 'sequence' column and no 'heavy_column' and 'light_column' were provided."
        )

    return dataset


def load_and_tokenize(
    data_files: dict,
    tokenizer: PreTrainedTokenizerBase,
    file_type: str = "parquet",
    heavy_column: str = None,
    light_column: str = None,
    separator: str = "<cls>",
    max_len: int = 256,
    truncate: bool = True,
    num_proc: int = 128,
    cache_dir: str = "~/.cache/huggingface/datasets",
):

    # load
    dataset = load_dataset(
        file_type, data_files=data_files, num_proc=num_proc, cache_dir=cache_dir
    )

    # get the first available split name
    split = next(iter(dataset.keys()))

    # if the 'sequence' column doesn't exist
    # use 'heavy_column' and 'light_column' to create it
    if "sequence" not in dataset[split].column_names:
        dataset = _generate_sequence(dataset, heavy_column, light_column, separator)

    # tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["sequence"],
            padding="max_length",
            max_length=max_len,
            truncation=truncate,
        ),
        batched=True,
        num_proc=num_proc,
        remove_columns=[
            col for col in dataset[split].column_names  # keep only tokenizer outputs
        ],
    )

    return tokenized_dataset
