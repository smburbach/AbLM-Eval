from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ["routing_compare"]

REGIONS = [
    "FRH1",
    "CDRH1",
    "FRH2",
    "CDRH2",
    "FRH3",
    "CDRH3",
    "FRH4",
    "FRL1",
    "CDRL1",
    "FRL2",
    "CDRL2",
    "FRL3",
    "CDRL3",
    "FRL4",
    "PAD",
]  # excluding BOS, SEP, & EOS


def _expert_balance(extracted, full_df):
    expert_balance = (
        extracted.groupby(["sequence_id", "layer", "expert_id"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    # fill experts not routed to with count 0 and remove tokens not routed
    expert_balance_df = pd.merge(
        full_df, expert_balance, on=["layer", "expert_id"], how="left"
    ).fillna({"count": 0})

    # calculate average # of tokens routed to each expert across layers
    mean_balance = expert_balance_df.groupby("layer")["count"].mean().mean()
    return expert_balance_df, mean_balance


def _compute_region_heatmap_data(region, extracted, full_df, expert_balance_df):
    region_counts = (
        extracted[extracted["region"] == region]
        .groupby(["sequence_id", "layer", "expert_id"])
        .size()
        .reset_index(name="count")
    )

    pad_filled = full_df.merge(
        region_counts, on=["layer", "expert_id"], how="left"
    ).fillna({"count": 0})

    pad_merged = expert_balance_df.merge(
        pad_filled,
        on=["sequence_id", "layer", "expert_id"],
        how="left",
        suffixes=("_total", "_pad"),
    )
    pad_merged["count_pad"] = pad_merged["count_pad"].fillna(0)

    pad_merged["pad_percentage"] = (
        pad_merged["count_pad"] / pad_merged["count_total"] * 100
    )

    heatmap_data = pad_merged.pivot_table(
        index="layer", columns="expert_id", values="pad_percentage", aggfunc="mean"
    )

    # calculate means
    grouped = pad_merged.groupby("layer")["pad_percentage"]
    per_layer_means = grouped.mean()
    per_layer_stderr = grouped.sem()
    mean_region = per_layer_means.mean()

    return heatmap_data, mean_region, per_layer_means, per_layer_stderr


def _multi_region_heatmap(
    extracted, full_df, expert_balance_df, model_name, output_dir
):
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 20))
    axes = axes.flatten()

    all_region_means = {}
    all_layer_means = {}
    all_layer_stderr = {}

    for i, region in enumerate(REGIONS):
        # process data
        heatmap_data, mean_region, per_layer_means, per_layer_stderr = (
            _compute_region_heatmap_data(region, extracted, full_df, expert_balance_df)
        )

        # plot
        sns.heatmap(
            heatmap_data, annot=False, fmt=".1f", cmap="Blues", cbar=True, ax=axes[i]
        )
        axes[i].set_title(f"{region} (mean={mean_region:.1f}%)")
        axes[i].set_xlabel("Expert ID")
        axes[i].set_ylabel("Layer")

        # save means
        all_region_means[region] = mean_region
        all_layer_means[region] = per_layer_means
        all_layer_stderr[region] = per_layer_stderr

    # save fig
    plt.tight_layout()
    plt.savefig(
        f"./{output_dir}/{model_name}_region-heatmaps.png",
        bbox_inches="tight",
        dpi=300,
    )

    return all_region_means, all_layer_means, all_layer_stderr


def _plot_region_layer_means(summary_df, region, output_dir):
    # sort
    models = summary_df["model"].tolist()

    # find the max number of layers across all models
    max_layers = max(len(row) for row in summary_df[f"{region}_layer_means"])

    # initialize the x-axis with layers from 0 to max_layers
    layers = np.arange(max_layers)
    width = 0.8 / len(models)  # adjust bar width based on the number of models

    fig, ax = plt.subplots(figsize=(12, 6))

    # loop through models
    for i, model in enumerate(models):
        means = summary_df[f"{region}_layer_means"].iloc[i]
        stderr = summary_df[f"{region}_layer_stderr"].iloc[i]

        # pad if there are fewer layers for this model
        padded_means = np.pad(
            means, (0, max_layers - len(means)), mode="constant", constant_values=np.nan
        )
        padded_stderr = np.pad(
            stderr, (0, max_layers - len(stderr)), mode="constant", constant_values=0.0
        )

        # plot
        ax.bar(
            layers + i * width,  # offset for each model
            padded_means,
            yerr=padded_stderr,
            width=width,
            label=model,
        )

    # labeling and formatting
    ax.set_ylabel(f"{region} Mean % Routing")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"Layer {i}" for i in layers])
    ax.legend(title="Model")

    # save fig
    plt.tight_layout()
    plt.savefig(
        f"./{output_dir}/{region}-compare.png",
        bbox_inches="tight",
        dpi=300,
    )


def routing_compare(results_dir, output_dir, **kwargs):

    summary_rows = []

    for file in Path(results_dir).glob("*routing_results.parquet"):
        df = pd.read_parquet(file)
        model_name = df["model"].iloc[0]

        num_layers = df["layer"].nunique()
        num_experts = df["expert_id"].dropna().nunique()
        full_df = pd.DataFrame(
            [(l, e) for l in range(num_layers) for e in range(num_experts)],
            columns=["layer", "expert_id"],
        )

        expert_balance_df, mean_balance = _expert_balance(df, full_df)
        region_means, layer_means, layer_stderr = _multi_region_heatmap(
            df, full_df, expert_balance_df, model_name, output_dir
        )

        row = {
            "model": model_name,
            "mean_balance": mean_balance,
        }

        # Add region means as individual columns
        for region, val in region_means.items():
            row[f"{region}_mean"] = val

        # Add layer means as lists per region
        for region, layers in layer_means.items():
            row[f"{region}_layer_means"] = [
                layers.get(i, None) for i in range(num_layers)
            ]

        # Add layer std err as lists per region
        for region, layers in layer_stderr.items():
            row[f"{region}_layer_stderr"] = [
                layers.get(i, None) for i in range(num_layers)
            ]

        summary_rows.append(row)

    # convert to df & sort
    df = pd.DataFrame(summary_rows)
    df_sorted = df.sort_values("model")

    # plot CDRH3 summary
    _plot_region_layer_means(df_sorted, region="CDRH3", output_dir=output_dir)

    # save
    df_sorted.to_csv(f"{output_dir}/routing_summary.csv", index=False)
