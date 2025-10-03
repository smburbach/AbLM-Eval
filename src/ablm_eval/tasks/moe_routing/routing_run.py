from tqdm import tqdm
import pandas as pd
import torch
import re

from ...utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
    move_to_cpu,
)
from .routing_config import RoutingConfig

__all__ = ["run_routing_analysis"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_routing_analysis(model_name: str, model_path: str, config: RoutingConfig):

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path, tokenizer_path=config.tokenizer_path, task="mlm"
    )
    model = model.to(device)
    model.eval()

    # load & process dataset
    tokenized_dataset = load_and_tokenize(
        data_path=config.data_path,
        tokenizer=tokenizer,
        config=config,
    )

    # inference
    outputs = _inference(model, tokenized_dataset)

    # append outputs to original dataset
    data = tokenized_dataset.to_pandas()
    data["balmmoe_output"] = outputs

    # process outputs
    extracted = _process_outputs(data, config, tokenizer)
    extracted["model"] = model_name

    # save results
    data["balmmoe_output"] = data["balmmoe_output"].apply(_tensor_to_python)
    data.to_parquet(f"{config.output_dir}/results/{model_name}_raw-outputs.parquet")
    extracted.to_parquet(
        f"{config.output_dir}/results/{model_name}_routing_results.parquet"
    )


def _parse_regions(
    chains,
    max_length,
    label_map: dict,
    tokens: list[str],
    special_tokens: set[str],
):
    """
    Parse CDR masks to generation position:name mapping.
    Expects CDR masks to label FR regions with 0 and CDR regions with 1.
    """

    # process each chain
    labels = []
    for i, chain in enumerate(chains):
        count = {k: 1 for k in label_map}
        prev_char = None

        # loop through mask
        for char in chain["mask"]:
            # new region
            if char != prev_char:
                region = label_map[char]
                if region.startswith("CDR") and len(region) == 4:  # ex. "CDR1"
                    label = f"CDR{chain['chain_name']}{region[-1]}"  # ex. convert to "CDRH1"
                else:
                    label = f"{region}{chain['chain_name']}{count[char]}"
                count[char] += 1

            # append
            labels.append(label)
            prev_char = char

    # assign regions
    regions = {}
    ptr = 0
    for pos in range(max_length):
        if tokens[pos] in special_tokens:
            regions[pos] = tokens[pos]
        else:
            regions[pos] = labels[ptr]
            ptr += 1
    return regions


def _clean_special(tok: str) -> str:
    return re.sub(r"^<([^<>]+)>$", r"\1", tok).upper()


def _process_outputs(test_data: pd.DataFrame, config: RoutingConfig, tokenizer):
    data = []
    max_len = config.max_len
    chain_names = config.dataset_columns.chain_names
    cdr_cols = config.dataset_columns.cdr_columns
    locus_col = config.dataset_columns.locus_column

    mapping = (
        ("HEAVY", "H"),
        ("LIGHT", "L"),
        ("KAPPA", "L"),
        ("IGH", "H"),
        ("IGL", "L"),
        ("IGK", "L"),
    )
    special_tokens = {_clean_special(t) for t in tokenizer.all_special_tokens}

    for row in tqdm(
        test_data.itertuples(), total=len(test_data), desc="Processing outputs"
    ):

        sequence_id = getattr(row, config.dataset_columns.id_column)

        # sequence
        ids = getattr(row, "input_ids")
        tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
        tokens = [_clean_special(tok) for tok in tokens]

        # skip sequences where sequence length != cdr mask length
        non_special_count = sum(1 for t in tokens if t not in special_tokens)
        mask_len = sum(len(getattr(row, cdr_cols[i])) for i in range(len(chain_names)))
        if mask_len != non_special_count:
            continue

        # map cdr regions
        chains = []
        label_chars = set()
        for i, name in enumerate(chain_names):
            key = (
                name
                if config.antibody_datatype == "paired"
                else getattr(row, locus_col)
            ).upper()
            chain_label = next((v for p, v in mapping if p in key), "")

            # append info
            mask = getattr(row, cdr_cols[i])
            chains.append({"chain_name": chain_label, "mask": mask})
            label_chars.update(mask)

        # set label_map based on mask characters
        if {"2", "3"}.intersection(label_chars):
            label_map = {"0": "FR", "1": "CDR1", "2": "CDR2", "3": "CDR3"}
        else:
            label_map = {"0": "FR", "1": "CDR"}

        # map regions
        region_map = _parse_regions(
            chains,
            max_length=max_len,
            label_map=label_map,
            tokens=tokens,
            special_tokens=special_tokens,
        )

        # extract
        for layer, expert_idxs in enumerate(row.balmmoe_output["expert_indexes"]):
            exp2pos = {
                eid: set(idxs[idxs != -1].tolist())
                for eid, idxs in enumerate(expert_idxs)
            }

            pos2exp = {}
            for eid, pos_set in exp2pos.items():
                for p in pos_set:
                    pos2exp.setdefault(p, []).append(eid)

            for pos in range(max_len):
                experts = pos2exp.get(
                    pos, [pd.NA]
                )  # NA if token is not sent to any expert
                for eid in experts:
                    data.append(
                        {
                            "sequence_id": sequence_id,
                            "layer": layer,
                            "expert_id": eid,
                            "token_position": pos,
                            "amino_acid": tokens[pos],
                            "region": region_map.get(pos, "Unknown"),
                        }
                    )

    return pd.DataFrame(data)


def _inference(model, tokenized_dataset) -> list:
    outputs = []
    for row in tqdm(tokenized_dataset, desc="Running inference"):
        # format model inputs
        input_ids = torch.tensor(row["input_ids"], device=device).unsqueeze(0)
        attention_mask = torch.tensor(row["attention_mask"], device=device).unsqueeze(0)

        with torch.no_grad():
            output = model(
                input_ids,
                labels=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_router_logits=True,
                output_expert_indexes=True,
            )
            outputs.append(move_to_cpu(output))
    return outputs


def _tensor_to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: _tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_tensor_to_python(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(_tensor_to_python(i) for i in obj)
    return obj
