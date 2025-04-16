import os
from typing import Union

from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = ["load_and_tokenize"]


def _generate_sequence(dataset, heavy_column, light_column, separator):
    """
    Handle the logic of combining heavy_column and light_column into 'sequence' if necessary.
    """
    if heavy_column and light_column:
        if (
            heavy_column in dataset.column_names
            and light_column in dataset.column_names
        ):
            # concat heavy_column and light_column to create the 'sequence' column
            dataset = dataset.map(
                lambda x: {"sequence": "".join(x[heavy_column]) + separator + "".join(x[light_column])},
            )
        else:
            raise ValueError(
                f"Both columns {heavy_column} and {light_column} must exist in the dataset."
            )
    elif "sequence" not in dataset.column_names:
        raise ValueError(
            "The dataset does not contain a 'sequence' column and no 'heavy_column' and 'light_column' were provided."
        )

    return dataset


def load_and_tokenize(
    data_path: str,
    tokenizer: PreTrainedTokenizerBase,
    config: EvalConfig
):

    # load
    file_type = os.path.splitext(data_path)[1][1:]
    dataset = load_dataset(
        file_type, data_files=data_path, split="train", num_proc=config.num_proc,
    )

    # use 'heavy_column' and 'light_column' to create 'sequence' column
    # if not provided, use the'sequence' column if it exists
    # otherwise, throw error
    dataset = _generate_sequence(dataset, config.heavy_column, config.light_column, config.separator)

    # tokenize
    drop_cols = [col for col in dataset.column_names]
    if config.return_sequence:
        drop_cols.remove("sequence")

    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["sequence"],
            padding=config.padding,
            max_length=config.max_len,
            truncation=config.truncate,
            add_special_tokens=config.add_special_tokens,
        ),
        batched=True,
        num_proc=config.num_proc,
        remove_columns=drop_cols,
    )

    return tokenized_dataset
