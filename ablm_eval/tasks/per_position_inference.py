from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
from tqdm import tqdm

from ..utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
)
from ..configs import PerPositionConfig, MutationPredConfig

__all__ = ["run_per_pos"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _inference_batched(model, tokenizer, input_ids):
    seq_len = input_ids.shape[0]

    # create a batch of inputs with one position masked at a time
    masked_inputs = input_ids.repeat(seq_len, 1)
    labels = torch.full_like(masked_inputs, -100)

    # mask positions diagonally, one at a time
    diagonal_idxs = torch.arange(seq_len)
    masked_inputs[diagonal_idxs, diagonal_idxs] = tokenizer.mask_token_id
    labels[diagonal_idxs, diagonal_idxs] = input_ids

    # send to device
    masked_inputs = masked_inputs.to(device)
    labels = labels.to(device)

    # inference
    with torch.no_grad():
        outputs = model(input_ids=masked_inputs, labels=labels)
        logits = outputs.logits

        # calculate loss and perplexity
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        ce_loss = ce_loss.view(seq_len, -1).sum(dim=1)
        ppl = torch.exp(ce_loss)

        # get predictions
        masked_logits = logits[diagonal_idxs, diagonal_idxs]
        probs = masked_logits.softmax(dim=-1)
        pred_tokens = masked_logits.argmax(dim=-1)
        pred_strings = [tokenizer.decode([t]) for t in pred_tokens]

    return {
        "loss": ce_loss.tolist(),
        "perplexity": ppl.tolist(),
        "probabilities": probs.tolist(),
        "prediction": pred_strings,
    }


def run_per_pos(
    model_name: str,
    model_path: str,
    config: Union[PerPositionConfig, MutationPredConfig],
):

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, task="mlm")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    # load & process dataset
    tokenized_dataset = load_and_tokenize(
        data_path=config.data_path, tokenizer=tokenizer, config=config
    )

    # inference
    results = []
    for example in tqdm(tokenized_dataset, desc="Running inference"):
        result = _inference_batched(
            model,
            tokenizer,
            input_ids=torch.tensor(example["input_ids"]),
        )
        # merge results with reference df
        combined = {
            "model_name": model_name,
            **{
                k: v
                for k, v in example.items()
                if k not in ("input_ids", "attention_mask")
            },
            **result,
        }
        results.append(combined)

    # save results
    df = pl.DataFrame(results)
    df.write_parquet(
        f"{config.output_dir}/results/{model_name}_{config.task_dir}.parquet"
    )
