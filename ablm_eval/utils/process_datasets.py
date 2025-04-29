import os
from typing import Union

from datasets import Dataset, DatasetDict, load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = ["load_and_tokenize"]


def _generate_sequence(dataset, column_names, config):
    """
    Handle the logic of combining heavy_column and light_column into 'sequence' if necessary.
    """
    seq_column = config.sequence_column
    h_column = config.heavy_column
    l_column = config.light_column

    if (h_column and l_column) and not seq_column:
        if (h_column in column_names) and (l_column in column_names):
            # concat heavy and light sequences to create 'sequence' column
            dataset = dataset.map(
                lambda x: {
                    "sequence": "".join(x[h_column])
                    + config.separator
                    + "".join(x[l_column])
                },
            )
            config.sequence_column = "sequence"
        else:
            raise ValueError(
                f"Both columns {h_column} and {l_column} must exist in the dataset."
            )
    elif seq_column and not (h_column or l_column):
        if seq_column not in column_names:
            raise ValueError(f"The column {seq_column} must exist in the dataset.")
    else:
        raise ValueError(
            "Please provide either the 'sequence_column' or both the 'heavy_column' and 'light_column'."
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

    # format sequence column if needed
    columns = dataset[key].column_names
    dataset = _generate_sequence(dataset, column_names=columns, config=config)

    # determine columns to drop
    drop_cols = [col for col in columns if col not in config.keep_columns]

    # tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x[config.sequence_column],
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
