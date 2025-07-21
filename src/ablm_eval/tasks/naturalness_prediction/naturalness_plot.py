import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ...utils.tables import table_compare

__all__ = ["naturalness_compare"]


# def naturalness_compare(results_dir, output_dir, task_str, **kwargs):

#     # save combined csv
#     df = table_compare(results_dir, output_dir, task_str, return_raw_data=True)

#     # sort by dataset
#     df["dataset"] = pd.Categorical(
#         df["dataset"], categories=sorted(df["dataset"].unique()), ordered=True
#     )
#     df = df.sort_values("dataset")

#     # plot
#     models = df["model"].unique()
#     datasets = df["dataset"].unique()
#     if len(datasets) == 1:
#         _plot_naturalness(
#             df, hue="model", plot_name="compare-models", output_dir=output_dir
#         )
#     else:
#         for model in models:
#             model_df = df[df["model"] == model].reset_index(drop=True)
#             _plot_naturalness(
#                 model_df, hue="dataset", plot_name=model, output_dir=output_dir
#             )

def naturalness_compare(results_dir, output_dir, task_str, **kwargs):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # save combined csv
    df = table_compare(results_dir, output_dir, task_str, return_raw_data=True)

    # sort by dataset
    df["dataset"] = pd.Categorical(
        df["dataset"], categories=sorted(df["dataset"].unique()), ordered=True
    )
    df["model"] = pd.Categorical(
        df["model"], categories=sorted(df["model"].unique()), ordered=True
    )
    df = df.sort_values(["dataset", "model"])

    # single combined plot
    _plot_naturalness(
        df,
        x="model",
        hue="dataset",
        plot_name="compare-naturalness",
        output_dir=output_dir,
    )


def _plot_naturalness(df, x, hue, plot_name, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    num_x = df[x].nunique()
    plt.figure(figsize=(max(6, num_x * 1.5), 5))

    sns.boxenplot(
        data=df,
        x=x,
        y="naturalness",
        hue=hue,
        dodge=True,
        showfliers=False,
        k_depth="proportion",
        outlier_prop=0.1,
        width=0.7,
        saturation=1,
    )

    # labels & ticks
    plt.xlabel(x.capitalize())
    plt.ylabel("Naturalness")
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="best")

    # save
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/{plot_name}.png",
        bbox_inches="tight",
        dpi=300,
    )


# def _plot_naturalness(df, hue, plot_name, output_dir):

#     # plot
#     plt.figure(figsize=(6, 4))
#     sns.boxenplot(
#         data=df,
#         x="naturalness",
#         hue=hue,
#         dodge=True,
#         showfliers=False,
#         k_depth="proportion",
#         outlier_prop=0.1,
#         width=0.7,
#         saturation=1,
#     )

#     # labels & ticks
#     plt.xlabel("Naturalness")
#     plt.yticks([])

#     # save
#     plt.tight_layout()
#     plt.savefig(
#         f"./{output_dir}/{plot_name}-naturalness.png",
#         bbox_inches="tight",
#         dpi=300,
#     )
