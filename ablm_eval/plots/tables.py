from pathlib import Path

import pandas as pd

__all__ = ["table_compare"]


def _combine_stats(paths):
    # read all results files and combine
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    # drop 'itr' col from classification tasks
    if "itr" in df.columns:
        df = df.drop(columns="itr")

    # get means and errors
    means = df.groupby("model").mean(numeric_only=True)
    sems = df.groupby("model").sem(numeric_only=True)

    # exclude sem if it is None
    def format_value(mean, sem):
        if pd.notna(sem):
            return f"{mean:.4f} (± {sem:.4f})"
        return f"{mean:.4f}"

    # combine in a readable table format
    combined = means.copy()
    for col in means.columns:
        combined[col] = means[col].combine(sems[col], format_value)

    return combined


def table_compare(config, **kwargs):
    # combine raw results files
    results_dir = Path(config.output_dir) / "results"
    files = list(results_dir.glob("*.csv"))
    combined_df = _combine_stats(files)

    # save
    combined_df.to_csv(f"{config.output_dir}/combined-{config.task_dir}-results.csv")
