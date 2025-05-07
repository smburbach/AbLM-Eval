import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..configs import NaturalnessConfig
from ..utils import load_reference_data
from .per_position_inference import run_per_pos

__all__ = ["run_naturalness"]


def _mean_log_prob(row, dataset):

    seq_log_probs = []
    for probs, germ_tok in zip(row.probabilities, row.tokenized_sequence):
        probs = torch.tensor(probs)
        log_probs = torch.log(probs)

        # extract prob of the wild-type residue
        wt_log_prob = log_probs[germ_tok].item()
        seq_log_probs.append(wt_log_prob)

    # calculate metrics across sequence
    mean_log_prob = np.mean(seq_log_probs)
    pseudo_ppl = np.exp(-mean_log_prob)
    naturalness = 1 / pseudo_ppl

    return pd.Series(
        {
            "model": row.model,
            "dataset": dataset.replace("-", ""),
            "sequence_id": row.sequence_id,
            "log_likelihood": mean_log_prob,
            "pseudo_perplexity": pseudo_ppl,
            "naturalness": naturalness,
        }
    )


def run_naturalness(model_name: str, model_path: str, config: NaturalnessConfig):

    # run per position inference
    run_per_pos(model_name, model_path, config)

    # load per position inference results
    data_name = f"{config.dataset_name}-" if config.dataset_name is not None else ""
    results = load_reference_data(
        f"{config.output_dir}/results/{model_name}_{data_name}per-position-inference.parquet"
    )

    # process
    df = results.apply(_mean_log_prob, dataset=data_name, axis=1)

    # save processed results
    df.to_parquet(
        f"{config.output_dir}/results/{model_name}_{data_name}naturalness.parquet"
    )
