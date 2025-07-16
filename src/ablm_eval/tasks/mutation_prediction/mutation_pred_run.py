import abutils
import pandas as pd
import numpy as np
from tqdm import tqdm

from .mutation_pred_config import MutationPredConfig
from ...utils import load_reference_data
from ..per_position_inference import run_per_pos

__all__ = ["run_mutation_pred"]


def run_mutation_pred(model_name: str, model_path: str, config: MutationPredConfig):

    # process data for mutation pred
    if not config.data_processed:
        _mutation_preprocessing(config)

    # run per position inference on processed data
    run_per_pos(model_name, model_path, config)

    # load & process per position inference results
    data_name = f"{config.dataset_name}-" if config.dataset_name is not None else ""
    results = load_reference_data(
        f"{config.output_dir}/results/{model_name}_{data_name}per-position-inference.parquet"
    )
    df = _process_per_pos_results(results, config.separator)

    # save processed results
    df.to_parquet(
        f"{config.output_dir}/results/{model_name}_{data_name}mutation-analysis.parquet"
    )


def _mutation_preprocessing(config):

    dataset_columns = config.dataset_columns

    # add alignments to keep columns
    chain_names = (
        dataset_columns.chain_names if config.antibody_datatype == "paired" else [None]
    )
    keep_columns = [dataset_columns.id_column] + dataset_columns.mutation_columns
    for chain in chain_names:
        prefix = f"_{chain}" if chain else ""
        keep_columns.extend(
            [
                f"sequence_alignment{prefix}",
                f"germline_alignment{prefix}",
            ]
        )

    # load reference data
    sep = config.separator
    df = load_reference_data(config.data_path, keep_columns=keep_columns)

    # filter for mutated sequences only
    mutation_columns = dataset_columns.mutation_columns
    df = df[np.logical_or.reduce([df[col] > 0 for col in mutation_columns])]

    data = []
    for row in tqdm(df.itertuples(), total=len(df), desc="Pre-processing data"):

        chain_seqs = []
        chain_germs = []
        for chain in chain_names:
            prefix = f"_{chain}" if chain else ""

            # translate chain
            seq_aa = abutils.tl.translate(getattr(row, f"sequence_alignment{prefix}"))
            germ_aa = abutils.tl.translate(getattr(row, f"germline_alignment{prefix}"))

            chain_seqs.append(seq_aa)
            chain_germs.append(germ_aa)

        sequence_mutated = sep.join(chain_seqs)
        sequence_germ = sep.join(chain_germs)

        assert len(sequence_mutated) == len(sequence_germ)

        data.append(
            {
                "sequence_id": row.sequence_id,
                "sequence_mutated": sequence_mutated,
                "sequence_germ": sequence_germ,
            }
        )
    data_df = pd.DataFrame(data)

    # save processed data
    data_path = f"{config.output_dir}/processed-data.parquet"
    data_df.to_parquet(data_path)

    # update config for future models using the same config
    config.data_path = data_path
    config.data_processed = True
    config.dataset_columns.chain_columns = [config.sequence_column]
    config.keep_columns.extend(["sequence_mutated", "sequence_germ"])


def _process_per_pos_results(results: pd.DataFrame, separator: str):

    results = results.apply(_analyze_row, separator=separator, axis=1)

    cols = [
        "positions",
        "mutated_aa",
        "germline_aa",
        "germ_probs",
        "pred_aa",
        "pred_probs",
        "predicted_germ",
    ]
    results = results.explode(cols)

    # position match
    # if it predicted a mutation, is there actually a mutation in this location?
    results["correct_position"] = (results["predicted_germ"] == False) & (
        results["mutated_aa"] != results["germline_aa"]
    )

    # chemistry match
    # if it predicted a mutation (and there is a mutation), is it a chemical match?
    results["correct_chemistry"] = results.apply(
        lambda row: (row["correct_position"])
        & (row["pred_aa"] in get_aa_group(row["mutated_aa"])),
        axis=1,
    )

    # amino acid match
    # if it predicted a mutation (and there is a mutation), is it the right amino acid?
    results["correct_amino_acid"] = (results["correct_position"] == True) & (
        results["mutated_aa"] == results["pred_aa"]
    )

    return results


def _analyze_row(row, separator: str):

    # convert to lists
    germline_aa = list(row.sequence_germ.replace(separator, ""))
    mutated_aa = list(row.sequence_mutated.replace(separator, ""))
    assert len(germline_aa) == len(mutated_aa)
    positions = list(range(len(germline_aa)))

    # calculate number of mutations
    num_mutations = sum((a != b) for a, b in zip(germline_aa, mutated_aa))

    # extract probabilities
    germ_probs, pred_probs, predicted_germs = [], [], []
    for germ, germ_tok, pred_tok, probs in zip(
        germline_aa,
        row.tokenized_seq_wo_special,
        row.prediction_tokens,
        row.probabilities,
    ):
        germ_probs.append(probs[germ_tok])
        pred_probs.append(probs[pred_tok])
        predicted_germs.append(germ_tok == pred_tok)

    # return row
    return pd.Series(
        {
            "model": row.model,
            "sequence_id": row.sequence_id,
            "total_mutations": num_mutations,
            "positions": positions,
            "mutated_aa": mutated_aa,
            "germline_aa": germline_aa,
            "germ_probs": germ_probs,
            "pred_aa": list(row.prediction),
            "pred_probs": pred_probs,
            "predicted_germ": predicted_germs,
        }
    )


AA_CHEM = [
    ["A", "G", "I", "L", "M", "V"],
    ["C", "S", "T", "P", "N", "Q"],
    ["D", "E"],
    ["K", "R", "H"],
    ["F", "Y", "W"],
]


def get_aa_group(aa):
    return next((group for group in AA_CHEM if aa in group), [])
