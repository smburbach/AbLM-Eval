from tqdm import tqdm
import pandas as pd
import torch

from ...utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
    move_to_cpu,
)
from .routing_config import RoutingConfig

__all__ = ["run_routing_analysis"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_regions(
    chains,
    max_length,
    label_map: dict = {"0": "FR", "1": "CDR"},
):
    """
    Parse CDR masks to generation position:name mapping.
    Expects CDR masks to label FR regions with 0 and CDR regions with 1.
    """

    regions, region_lengths = {}, {}
    pos = 0

    # helper to add single token to dicts
    def add_token(label):
        nonlocal pos
        regions[pos] = label
        region_lengths[label] = region_lengths.get(label, 0) + 1
        pos += 1

    # BOS
    add_token("BOS")

    # process each chain
    for i, chain in enumerate(chains):
        count = {k: 1 for k in label_map}
        prev_char = None

        # loop through mask
        for char in chain["mask"]:
            # new region
            if char != prev_char:
                label = f"{label_map[char]}{chain['prefix']}{count[char]}"
                count[char] += 1

            add_token(label)
            prev_char = char

        # SEP between chains
        if i < len(chains) - 1:
            add_token("SEP")

    # EOS
    add_token("EOS")

    # PAD to max_length
    pad_count = max_length - pos
    for i in range(pad_count):
        add_token("PAD")

    return regions, region_lengths


def _process_outputs(test_data, config: RoutingConfig):

    data, lengths = [], []
    for row in tqdm(
        test_data.itertuples(), total=len(test_data), desc="Processing outputs"
    ):

        sequence_id = getattr(row, config.id_column)

        # map cdr regions
        region_map, region_lengths = _parse_regions(
            chains=[
                {"prefix": "H", "mask": getattr(row, config.heavy_cdr_column)},
                {"prefix": "L", "mask": getattr(row, config.light_cdr_column)},
            ],
            max_length=config.max_len,
        )

        # sequence
        seq = list(
            "X"
            + getattr(row, config.heavy_column)
            + "X"
            + getattr(row, config.heavy_column)
        )
        if len(seq) < config.max_len:
            seq.extend(["X"] * (config.max_len - len(seq)))
        else:
            seq = seq[: config.max_len]

        # length data
        lengths.append({"sequence_id": sequence_id, **region_lengths})

        # extract data
        for layer, expert_idxs in enumerate(row.balmmoe_output["expert_indexes"]):
            expert_to_positions = {
                eid: set(idxs[idxs != -1].tolist())
                for eid, idxs in enumerate(expert_idxs)
            }

            position_to_experts = {}
            for eid, positions in expert_to_positions.items():
                for pos in positions:
                    position_to_experts.setdefault(pos, []).append(eid)

            for pos in range(config.max_len):
                experts = position_to_experts.get(
                    pos, [pd.NA]
                )  # NA if token is not sent to any expert
                for expert_id in experts:
                    data.append(
                        {
                            "sequence_id": sequence_id,
                            "layer": layer,
                            "expert_id": expert_id,
                            "token_position": pos,
                            "amino_acid": seq[pos],
                            "region": region_map.get(pos, "Unknown"),
                        }
                    )

    return pd.DataFrame(data), pd.DataFrame(lengths)


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
    else:
        return obj

def run_routing_analysis(model_name: str, model_path: str, config: RoutingConfig):

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, task="mlm")
    model = model.to(device)
    model.eval()

    # update keep_columns
    (config.keep_columns).extend(
        [
            config.id_column,
            config.heavy_column,
            config.light_column,
            config.heavy_cdr_column,
            config.light_cdr_column,
        ]
    )

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
    extracted, length_reference = _process_outputs(data, config)
    extracted["model"] = model_name
    length_reference["model"] = model_name

    # save raw results
    data["balmmoe_output"] = data["balmmoe_output"].apply(_tensor_to_python)
    data.to_parquet(
        f"{config.output_dir}/results/{model_name}_raw-outputs.parquet"
    )
    # save processed results
    extracted.to_parquet(
        f"{config.output_dir}/results/{model_name}_routing_results.parquet"
    )
    length_reference.to_parquet(
        f"{config.output_dir}/results/{model_name}_routing_length-reference.parquet"
    )
