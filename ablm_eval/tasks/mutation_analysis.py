import torch
import abutils
import polars as pl

from ..configs import MutationPredConfig
from ..utils import load_reference_data
from .per_position_inference import run_per_pos

__all__ = ["run_mutation_analysis"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mutation_preprocessing(config):
    df = load_reference_data(
        config.data_path,
        keep_columns=[
            "sequence_id",
            "v_mutation_count_aa_heavy",
            "sequence_alignment_heavy",
            "germline_alignment_heavy",
            "v_mutation_count_aa_light",
            "sequence_alignment_light",
            "germline_alignment_light",
        ],
    )

    # filter for mutated sequences only
    df = df[
        (df["v_mutation_count_aa_heavy"] > 0) | (df["v_mutation_count_aa_light"] > 0)
    ]

    data = []
    for row in df.itertuples():
        # translate heavy chain
        hseq = abutils.tl.translate(row.sequence_alignment_heavy)
        hgerm = abutils.tl.translate(row.germline_alignment_heavy)

        # translate light chain
        lseq = abutils.tl.translate(row.sequence_alignment_light)
        lgerm = abutils.tl.translate(row.germline_alignment_light)

        # combine heavy and light chains
        germ_aa = f"{hgerm}{config.separator}{lgerm}"
        seq_aa = f"{hseq}{config.separator}{lseq}"

        data.append(
            {
                "sequence_id": row.sequence_id,
                "v_mutation_count_aa_heavy": row.v_mutation_count_aa_heavy,
                "heavy_mutated": hseq,
                "heavy_germ": hgerm,
                "v_mutation_count_aa_light": row.v_mutation_count_aa_light,
                "light_mutated": lseq,
                "light_germ": lgerm,
                "sequence_mutated": seq_aa,
                "sequence_germ": germ_aa,
            }
        )
    data_df = pl.DataFrame(data)

    # save processed data
    data_path = f"{config.output_dir}/processed-data.parquet"
    data_df.write_parquet(data_path)

    # update config for future models using the same config
    config.data_path = data_path
    config.data_processed = True


def run_mutation_analysis(model_name: str, model_path: str, config: MutationPredConfig):

    # process data for mutation pred
    if not config.data_processed:
        _mutation_preprocessing(config)

    # run per position inference on processed data
    run_per_pos(model_name, model_path, config)
