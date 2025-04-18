import os
from typing import Union

from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = ["load_and_tokenize"]


def _generate_sequence(dataset, column_names, heavy_column, light_column, separator):
    """
    Handle the logic of combining heavy_column and light_column into 'sequence' if necessary.
    """

    if heavy_column and light_column:
        if (heavy_column in column_names) and (light_column in column_names):
            # concat heavy and light sequences to create 'sequence' column
            dataset = dataset.map(
                lambda x: {
                    "sequence": "".join(x[heavy_column])
                    + separator
                    + "".join(x[light_column])
                },
            )
        else:
            raise ValueError(
                f"Both columns {heavy_column} and {light_column} must exist in the dataset."
            )
    elif "sequence" not in column_names:
        raise ValueError(
            "The dataset does not contain a 'sequence' column and no 'heavy_column' and 'light_column' were provided."
        )

    return dataset


def load_and_tokenize(
    data_path: Union[str, dict], tokenizer: PreTrainedTokenizerBase, config
) -> Union[Dataset, DatasetDict]:

    # convert str to dict to simply logic
    return_dataset = False
    if isinstance(data_path, str):
        data_path = {"train": data_path}
        return_dataset = True

    key = next(iter(data_path))  # get name of first Dataset in DatasetDict

    # load
    file_type = os.path.splitext(data_path[key])[1][1:]
    dataset = load_dataset(
        file_type,
        data_files=data_path,
        num_proc=config.num_proc,
    )

    # use 'heavy_column' and 'light_column' to create 'sequence' column
    # if not provided, use the'sequence' column if it exists
    # otherwise, throw error
    columns = dataset[key].column_names
    dataset = _generate_sequence(
        dataset,
        column_names=columns,
        heavy_column=config.heavy_column,
        light_column=config.light_column,
        separator=config.separator,
    )

    # determine columns to drop
    # never drop label column
    drop_cols = [col for col in columns if col not in config.keep_columns]

    # tokenize
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

    # will return Dataset (not DatasetDict) if original path was a string
    return tokenized_dataset[key] if return_dataset else tokenized_dataset
