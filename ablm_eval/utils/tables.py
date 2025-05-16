from pathlib import Path

import pandas as pd

from .data import load_reference_data

__all__ = ["table_compare"]


def _combine_stats(paths):
    # read all results files and combine
    df = pd.concat([load_reference_data(path) for path in paths], ignore_index=True)

    # drop 'itr' col from classification tasks
    if "itr" in df.columns:
        df = df.drop(columns="itr")

    # get means and errors
    means = df.groupby(["model", "dataset"]).mean(numeric_only=True)
    sems = df.groupby(["model", "dataset"]).sem(numeric_only=True)

    # exclude sem if it is None
    def format_value(mean, sem):
        if pd.notna(sem):
            return f"{mean:.4f} (± {sem:.4f})"
        return f"{mean:.4f}"

    # combine in a readable table format
    combined = means.copy()
    for col in means.columns:
        combined[col] = means[col].combine(sems[col], format_value)

    return df, combined


def table_compare(
    results_dir, output_dir, task_str, return_raw_data: bool = False, **kwargs
):
    # combine raw results files
    extensions = [f"*{task_str}.csv", f"*{task_str}.parquet"]
    files = [f for ext in extensions for f in Path(results_dir).glob(ext)]
    raw_results, combined_df = _combine_stats(files)

    # save
    combined_df.to_csv(
        f"{output_dir}/combined-{task_str}-results.csv", 
        index=True # model name & dataset are the indices
    )

    if return_raw_data:
        return raw_results
