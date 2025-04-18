import os

from tqdm import tqdm
import pandas as pd
import polars as pl
import torch

from ..utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
)
from ..configs import RoutingConfig

__all__ = ["run_routing_analysis"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_chain(mask, prefix, pos_shift):
    regions, count_cdr, count_fr = {}, 1, 1
    prev_char, region_label = None, None
    region_lengths = {}

    for pos, char in enumerate(mask):
        if char == "1":
            if prev_char != "1":
                region_label = f"CDR{prefix}{count_cdr}"
                count_cdr += 1
        else:  # char == "0"
            if prev_char != "0":
                region_label = f"FR{prefix}{count_fr}"
                count_fr += 1
        regions[pos + pos_shift] = region_label
        prev_char = char

        region_lengths[region_label] = region_lengths.get(region_label, 0) + 1

    return regions, (pos + pos_shift), region_lengths


def _parse_cdr_mask(heavy_cdr_mask, light_cdr_mask, max_length=256):
    def _add_special_token(regions, region_lengths, token_name, position, length=1):
        regions[position] = token_name
        region_lengths[token_name] = length
        return regions, region_lengths

    # init dicts
    regions, region_lengths = {}, {}

    # BOS
    regions, region_lengths = _add_special_token(regions, region_lengths, "BOS", 0)

    # Heavy chain
    heavy_regions, last_heavy_pos, heavy_lengths = _parse_chain(heavy_cdr_mask, "H", 1)
    regions.update(heavy_regions)
    region_lengths.update(heavy_lengths)

    # SEP
    sep_pos = last_heavy_pos + 1
    regions, region_lengths = _add_special_token(
        regions, region_lengths, "SEP", sep_pos
    )

    # Parse light chain
    light_regions, last_light_pos, light_lengths = _parse_chain(
        light_cdr_mask, "L", sep_pos + 1
    )
    regions.update(light_regions)
    region_lengths.update(light_lengths)

    # Insert EOS token
    eos_pos = last_light_pos + 1
    regions, region_lengths = _add_special_token(
        regions, region_lengths, "EOS", eos_pos
    )

    # Assign PAD to remaining positions
    regions.update({pos: "PAD" for pos in range(eos_pos + 1, max_length)})
    region_lengths["PAD"] = max_length - eos_pos - 1

    return regions, region_lengths


def _process_outputs(test_data, config: RoutingConfig):

    data, lengths = [], []
    for p in test_data.itertuples():
        # map cdr regions
        sequence_id = getattr(p, config.id_column)
        region_map, region_lengths = _parse_cdr_mask(
            getattr(p, config.heavy_cdr_column),
            getattr(p, config.light_cdr_column),
            max_length=config.max_len,
        )

        # region length data
        lengths.append({"sequence_id": sequence_id, **region_lengths})

        # extract
        for layer in range(len(p.balmmoe_output["expert_indexes"])):
            expert_idxs = p.balmmoe_output["expert_indexes"][
                layer
            ]  # shape: (num_experts, expert_capacity)

            # selected tokens
            selected_tokens = set()
            for expert_id in range(expert_idxs.shape[0]):
                token_indices = expert_idxs[expert_id]  # Tokens assigned to this expert

                valid_tokens = token_indices[
                    token_indices != -1
                ].tolist()  # Ignore padding (-1s)

                for pos in valid_tokens:
                    selected_tokens.add(pos)
                    data.append(
                        {
                            "sequence_id": sequence_id,
                            "layer": layer,
                            "expert_id": expert_id,
                            "token_position": pos,
                            "region": region_map.get(pos, "Unknown"),
                        }
                    )

            # add tokens not selected by any expert in this layer
            for pos in range(256):
                if pos not in selected_tokens:
                    data.append(
                        {
                            "sequence_id": sequence_id,
                            "layer": layer,
                            "expert_id": "NA",  # Mark as not selected by any expert
                            "token_position": pos,
                            "region": region_map.get(pos, "Unknown"),
                        }
                    )

    # Convert list to DataFrame
    extracted = pl.DataFrame(data)
    length_reference = pl.DataFrame(lengths)

    return extracted, length_reference


def _move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_cpu(v) for v in obj)
    else:
        return obj


def _load_reference_data(path: str, keep_columns: list) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)[keep_columns]
    elif ext == ".csv":
        return pd.read_csv(path)[keep_columns]
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def run_routing_analysis(config: RoutingConfig):

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model_path, task="mlm")
    model = model.to(device)

    # update keep_columns
    (config.keep_columns).extend(
        [config.id_column, "sequence", config.heavy_cdr_column, config.light_cdr_column]
    )

    # load & process dataset
    tokenized_dataset = load_and_tokenize(
        data_path=config.routing_data,
        tokenizer=tokenizer,
        config=config,
    )

    # loop through dataset
    outputs = []
    for itr, p in enumerate(tqdm(tokenized_dataset)):
        if itr >= 10:
            break

        with torch.no_grad():
            input_ids = torch.tensor(p["input_ids"], device=device).unsqueeze(0)
            attention_mask = torch.tensor(p["attention_mask"], device=device).unsqueeze(
                0
            )
            output = model(
                input_ids,
                labels=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_router_logits=True,
                output_expert_indexes=True,
            )
            outputs.append(_move_to_cpu(output))

    # append outputs to original dataset
    ref = _load_reference_data(config.routing_data, keep_columns=config.keep_columns)[
        :10
    ]
    ref["balmmoe_output"] = outputs

    extracted, length_reference = _process_outputs(ref, config)

    # save results
    extracted.write_parquet(
        f"{config.output_dir}/{config.model_name}/routing_results.parquet"
    )
    length_reference.write_parquet(
        f"{config.output_dir}/{config.model_name}/routing_length-reference.parquet"
    )

    # TODO: plot results
