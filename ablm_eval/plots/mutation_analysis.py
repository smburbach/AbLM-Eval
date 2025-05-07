from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = ["mut_analysis_compare"]


def _filter_positions(dict, filter_column, filter_bool):
    positions = {}
    for model, df in dict.items():
        filtered = df[df[filter_column] == filter_bool]
        positions[model] = filtered["positions"].to_numpy()
    return positions


def _plot_histogram(
    position_data,
    bins=50,
    output_dir: str = None,
    plot_desc: str = None,
):
    """
    Plot side-by-side histograms of mutation positions.
    """
    model_names = list(position_data.keys())
    data_list = list(position_data.values())
    all_data = np.concatenate(data_list)
    bin_edges = np.histogram_bin_edges(all_data, bins=bins)
    bin_width = np.diff(bin_edges)[0]
    bar_width = bin_width / len(model_names)
    centers = bin_edges[:-1] + bin_width / 2

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(sorted(position_data.keys())):
        data = position_data[model]
        counts, _ = np.histogram(data, bins=bin_edges)
        offset = (i - (len(model_names) - 1) / 2) * bar_width
        positions = centers + offset
        plt.bar(positions, counts, width=bar_width, edgecolor="black", label=model)

    plt.title((plot_desc).replace("-", " ").title())
    plt.xlabel("Sequence Position")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"./{output_dir}/{plot_desc}_histogram.png",
        bbox_inches="tight",
        dpi=300,
    )


def _compute_match_ratios(data_dict):
    ratios = {}
    for model, df in data_dict.items():

        # mean matches per sequence
        grouped = df.groupby("sequence_id")[
            ["correct_position", "correct_chemistry", "correct_amino_acid"]
        ].sum()
        pos_match = grouped["correct_position"].mean()
        chem_match = grouped["correct_chemistry"].mean()
        aa_match = grouped["correct_amino_acid"].mean()

        # mean actual mutations per sequence
        mean_mut = df.drop_duplicates("sequence_id")["total_mutations"].mean()

        ratios[model] = {
            "position_match": pos_match,
            "chemistry_match": chem_match,
            "amino_acid_match": aa_match,
            "mean_mutations": mean_mut,
        }
    return ratios


def _plot_match_ratios(ratio_dict, output_dir: str = None):
    import matplotlib.pyplot as plt
    import numpy as np

    models = sorted(ratio_dict.keys())
    measurements = ["Position Match", "Chemical Match", "Amino Acid Match"]

    data = np.array(
        [
            [
                ratios["position_match"],
                ratios["chemistry_match"],
                ratios["amino_acid_match"],
            ]
            for ratios in ratio_dict.values()
        ]
    ).T

    bar_width = 0.8 / len(models)
    index = np.arange(len(measurements))

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(models):
        match_vals = data[:, i]
        bars = ax.bar(index + i * bar_width, match_vals, bar_width, label=model)
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{yval:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    # dashed line for mean mutations per sequence
    mean_mutations = ratio_dict[models[0]].get("mean_mutations", None)
    line = ax.axhline(
        y=mean_mutations,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean Mutations per Sequence",
    )

    # labels & ticks
    ax.set_ylabel("Average Per Sequence")
    ax.set_title("Predicted Mutation Matches Per Sequence")
    ax.set_xticks(index + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(measurements)

    # legend
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=10,
        title="Model",
    )

    plt.tight_layout()
    plt.savefig(
        f"./{output_dir}/mutation-match-ratios.png",
        bbox_inches="tight",
        dpi=300,
    )


def mut_analysis_compare(results_dir, output_dir, **kwargs):
    # load results
    data = {}
    for file in Path(results_dir).glob("*mutation-analysis.parquet"):
        df = pd.read_parquet(file)
        model_name = df["model"].iloc[0]
        data[model_name] = df

    # plot distribution of all predicted mutations
    all_mutations = _filter_positions(
        data, filter_column="predicted_germ", filter_bool=False
    )
    _plot_histogram(
        all_mutations, output_dir=output_dir, plot_desc="all-predicted-mutations"
    )

    # plot distribution of all correctly placed predicted mutations
    true_mutations = _filter_positions(
        data, filter_column="correct_position", filter_bool=True
    )
    _plot_histogram(
        true_mutations,
        output_dir=output_dir,
        plot_desc="true-positions-predicted-mutations",
    )

    # plot ratios of position, chemical and aa matches
    ratio_data = _compute_match_ratios(data)
    _plot_match_ratios(ratio_data, output_dir=output_dir)
