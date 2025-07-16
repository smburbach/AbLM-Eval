from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ["per_pos_compare"]


REGIONS = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]


def per_pos_compare(results_dir, output_dir, task_str, **kwargs):
    # load & concat results
    files = list(Path(results_dir).glob("*.parquet"))
    results = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)

    # return if CDR masks are not provided
    dataset_columns = results["dataset_columns"].iloc[0]
    if dataset_columns["cdr_columns"] is None:
        print("CDR columns not provided; skipping per-position comparison.")
        return

    # process results
    results = _extract(results, dataset_columns)
    data = _process_regions(results, dataset_columns)
    data_df = pd.DataFrame(data)

    # plots
    for mutated in sorted(data_df["mutated"].unique(), key=lambda x: (str(x))):
        df = data_df[(data_df["mutated"] == mutated)]
        for metric in ["median_loss", "accuracy"]:
            _per_pos_boxenplot(
                df,
                y_axis=metric,
                output_dir=output_dir,
                task_str=task_str,
                plot_desc=f"{mutated}_{metric}",
            )

    # summary df
    _summary_df(data_df, output_dir=output_dir, task_str=task_str)


def _extract(df: pd.DataFrame, dataset_columns: dict) -> pd.DataFrame:
    """Split concatenated sequence/loss/prediction into per-chain columns."""
    chains = dataset_columns["chain_names"]
    new_cols = {
        f"{chain}_{col}": [] for chain in chains for col in ["loss", "pred", "sequence"]
    }

    for _, r in df.iterrows():
        sep = r["separator"]
        seqs = [s for s in r["sequence"].split(sep) if s]
        losses = r["loss"]
        preds = r["prediction"]

        start = 0
        for chain, seq in zip(chains, seqs):
            # determine end idx based on chain length
            end = start + len(seq)

            # extract
            loss_slice = losses[start:end]
            pred_slice = preds[start:end]
            seq_list = list(seq)

            # length check
            assert len(loss_slice) == len(seq_list)
            assert len(pred_slice) == len(seq_list)

            # append results
            new_cols[f"{chain}_loss"].append(losses[start:end])
            new_cols[f"{chain}_pred"].append(preds[start:end])
            new_cols[f"{chain}_sequence"].append(list(seq))

            # update start idx
            start = end

    return df.assign(**new_cols)


def _process_regions(df: pd.DataFrame, dataset_columns: dict):
    """Processes each chain one at time, by region"""

    # get dataset columns
    chains = dataset_columns["chain_names"]
    cdr_cols = dataset_columns["cdr_columns"]
    mut_cols = dataset_columns["mutation_columns"]

    def get_mutation_status(mut_val):
        if mut_val is None or pd.isna(mut_val):
            return "mutation_unknown"
        return "mutated" if mut_val else "unmutated"

    data = []
    # loop through rows
    for _, row in df.iterrows():

        mutated_flags = [get_mutation_status(row.get(col, None)) for col in mut_cols]

        # loop through chains
        for i, chain in enumerate(chains):
            loss = row[f"{chain}_loss"]
            pred = row[f"{chain}_pred"]
            seq = row[f"{chain}_sequence"]
            cdr_mask = row[cdr_cols[i]]
            locus = (
                chain
                if row["antibody_datatype"] == "paired"
                else row.get(dataset_columns["locus_column"], chain)
            )

            # segment regions
            mask_segments = []
            prev_char = cdr_mask[0]
            start = 0
            for j, char in enumerate(cdr_mask):
                if char != prev_char:
                    mask_segments.append((start, j))
                    start = j
                prev_char = char
            mask_segments.append((start, len(cdr_mask)))

            # skip any sequences w/o 7 regions
            if len(mask_segments) != len(REGIONS):
                continue

            # extract by region
            for region, (start, end) in zip(REGIONS, mask_segments):
                region_loss = loss[start:end]
                region_pred = pred[start:end]
                region_seq = seq[start:end]
                data.append(
                    {
                        "region": region,
                        "model": row["model"],
                        "chain": locus,
                        "mutated": mutated_flags[i],
                        "loss": region_loss,
                        "median_loss": np.median(region_loss),
                        "accuracy": np.mean(
                            [p == t for p, t in zip(region_pred, region_seq)]
                        ),
                    }
                )
    return data


def _per_pos_boxenplot(
    df: pd.DataFrame,
    y_axis: str,
    output_dir: str,
    task_str: str,
    plot_desc: str,
):
    # chains & model order
    chains = sorted(df["chain"].unique())
    n_chains = len(chains)
    model_order = sorted(df["model"].unique())

    # create figure
    fig, axes = plt.subplots(n_chains, 1, figsize=(8, 3 * n_chains), sharex=True)
    if n_chains == 1:
        axes = [axes]

    for ax, chain in zip(axes, chains):
        # boxplot
        sns.boxenplot(
            data=df[df["chain"] == chain],
            x="region",
            y=y_axis,
            hue="model",
            hue_order=model_order,
            dodge=True,
            showfliers=False,
            k_depth="proportion",
            outlier_prop=0.1,
            width=0.7,
            saturation=1,
            ax=ax,
        )

        # ticks, labels, & legend
        ax.tick_params(axis="x", labelsize=11)
        ax.set_xlabel("")
        ax.set_ylabel(
            f"{chain.title()} Chain \n Per-position {y_axis.replace('_', ' ').title()}",
            fontsize=12,
        )
        ax.get_legend().remove()

    # legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=10,
        title="Model",
    )

    plt.tight_layout()
    plt.savefig(
        f"./{output_dir}/combined-{task_str}-results_{plot_desc}.png",
        bbox_inches="tight",
        dpi=300,
    )


def _summary_df(
    df: pd.DataFrame,
    output_dir: str,
    task_str: str,
):
    """Summary metrics for CDRH3 only."""
    # filter for CDR3 only
    cdr3_df = df[
        (df["region"] == "CDR3") & (df["chain"].str.lower().isin(["heavy", "h"]))
    ]
    if cdr3_df.empty:
        return

    # group by model, chain, mutated
    means = cdr3_df.groupby(["model", "mutated"]).median(numeric_only=True)
    sems = cdr3_df.groupby(["model", "mutated"]).sem(numeric_only=True)

    # format mean ± sem
    def format_value(mean, sem):
        return f"{mean:.4f} (± {sem:.4f})" if pd.notna(sem) else f"{mean:.4f}"

    # combine
    combined = pd.DataFrame(index=means.index)
    for col in means.columns:
        combined[f"CDRH3_{col}"] = means[col].combine(sems[col], format_value)

    # make model & mutated columns non-index cols, then sort
    combined = combined.reset_index()
    combined = combined.sort_values(by=["model", "mutated"])

    # save
    combined.to_csv(f"{output_dir}/results-summary_{task_str}.csv", index=False)
