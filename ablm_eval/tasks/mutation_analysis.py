import abutils
import pandas as pd
from tqdm import tqdm

from ..configs import MutationPredConfig
from ..utils import load_reference_data
from .per_position_inference import run_per_pos

__all__ = ["run_mutation_analysis"]


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
    for row in tqdm(df.itertuples(), total=len(df), desc="Pre-processing data"):
        # translate heavy chain
        hseq = abutils.tl.translate(row.sequence_alignment_heavy)
        hgerm = abutils.tl.translate(row.germline_alignment_heavy)

        # translate light chain
        lseq = abutils.tl.translate(row.sequence_alignment_light)
        lgerm = abutils.tl.translate(row.germline_alignment_light)

        # combine heavy and light chains
        germ_aa = f"{hgerm}{config.separator}{lgerm}"
        seq_aa = f"{hseq}{config.separator}{lseq}"
        assert len(germ_aa) == len(seq_aa)

        data.append(
            {
                "sequence_id": row.sequence_id,
                "v_mutation_count_aa_heavy": row.v_mutation_count_aa_heavy,
                "v_mutation_count_aa_light": row.v_mutation_count_aa_light,
                "sequence_mutated": seq_aa,
                "sequence_germ": germ_aa,
            }
        )
    data_df = pd.DataFrame(data)

    # save processed data
    data_path = f"{config.output_dir}/processed-data.parquet"
    data_df.to_parquet(data_path)

    # update config for future models using the same config
    config.data_path = data_path
    config.data_processed = True


def _analyze_row(row, separator: str):

    # convert to lists
    germline_aa = list(row.sequence_germ.replace(separator, "X"))
    mutated_aa = list(row.sequence_mutated.replace(separator, "X"))
    positions = list(range(len(germline_aa)))

    # calculate number of mutations
    num_mutations = sum(
        (a != b) for a, b in zip(germline_aa, mutated_aa) if a != "X" and b != "X"
    )

    # extract probabilities
    germ_probs, pred_probs, predicted_germs = [], [], []
    for germ, germ_tok, pred_tok, probs in zip(
        germline_aa,
        row.tokenized_sequence,
        row.prediction_tokens,
        row.probabilities,
    ):
        if germ == "X":  # ensures separator gets filtered out
            germ_probs.append(1.0)
            pred_probs.append(1.0)
            predicted_germs.append(True)
        else:
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
            "pred_aa": row.prediction,
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


def run_mutation_analysis(model_name: str, model_path: str, config: MutationPredConfig):

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
