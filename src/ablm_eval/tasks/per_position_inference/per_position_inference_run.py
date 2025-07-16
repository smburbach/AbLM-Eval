import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from ...utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
)

__all__ = ["run_per_pos"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _inference_batched(model, tokenizer, input_ids):

    # special tokens mask
    special_token_ids = set(tokenizer.all_special_ids)
    special_token_ids.remove(tokenizer.unk_token_id)  # keep unk token
    is_not_special = torch.tensor(
        [token_id.item() not in special_token_ids for token_id in input_ids],
        dtype=torch.bool,
    )
    valid_positions = torch.where(is_not_special)[0]
    num_valid = valid_positions.shape[0]

    # create a batch of inputs with one position masked at a time
    masked_inputs = input_ids.repeat(num_valid, 1)
    labels = torch.full_like(masked_inputs, -100)

    # mask positions diagonally, one at a time
    masked_inputs[range(num_valid), valid_positions] = tokenizer.mask_token_id
    labels[range(num_valid), valid_positions] = input_ids[valid_positions]

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
        ce_loss = ce_loss.view(num_valid, -1).sum(dim=1)
        ppl = torch.exp(ce_loss)

        # get predictions
        masked_logits = logits[range(num_valid), valid_positions]
        probs = masked_logits.softmax(dim=-1)
        pred_tokens = masked_logits.argmax(dim=-1)
        pred_strings = [tokenizer.decode([t]) for t in pred_tokens]

    return {
        "tokenized_sequence": input_ids.tolist(),
        "tokenized_seq_wo_special": [
            t.item() for t in input_ids if t.item() not in special_token_ids
        ],
        "loss": ce_loss.tolist(),
        "perplexity": ppl.tolist(),
        "probabilities": probs.tolist(),
        "prediction_tokens": pred_tokens.tolist(),
        "prediction": pred_strings,
    }


def run_per_pos(
    model_name: str,
    model_path: str,
    config,
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
            "model": model_name,
            "antibody_datatype": config.antibody_datatype,
            "dataset_columns": config.dataset_columns.to_dict(),
            "separator": config.separator,
            **{
                k: v
                for k, v in example.items()
                if k not in ("input_ids", "attention_mask")
            },
            **result,
        }
        results.append(combined)

    # save results
    df = pd.DataFrame(results)
    data_name = f"{config.dataset_name}-" if config.dataset_name is not None else ""
    df.to_parquet(
        f"{config.output_dir}/results/{model_name}_{data_name}per-position-inference.parquet"
    )
