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

# TODO: fix chains for unpaired sequences, using the locus_column


def _parse_regions(
    chains,
    max_length,
    label_map: dict = {"0": "FR", "1": "CDR"},
):
    """
    Parse CDR masks to generation position:name mapping.
    Expects CDR masks to label FR regions with 0 and CDR regions with 1.
    """

    regions = {}
    pos = 0

    # helper to add single token to dicts
    def add_token(label):
        nonlocal pos
        regions[pos] = label
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
                label = f"{label_map[char]}{chain['chain_name']}{count[char]}"
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

    return regions


def _process_outputs(test_data: pd.DataFrame, config: RoutingConfig):
    data = []
    max_len = config.max_len
    chain_names = config.dataset_columns.chain_names
    cdr_cols = config.dataset_columns.cdr_columns
    chain_cols = config.dataset_columns.chain_columns
    locus_col = config.dataset_columns.locus_column

    mapping = {"H": "H", "L": "L", "K": "L"}

    for row in tqdm(
        test_data.itertuples(), total=len(test_data), desc="Processing outputs"
    ):

        sequence_id = getattr(row, config.dataset_columns.id_column)

        # map cdr regions
        chains = []
        for i, name in enumerate(chain_names):
            key = (
                name[0].upper()
                if config.antibody_datatype == "paired"
                else getattr(row, locus_col)[i][0].upper()
            )
            chain_name = mapping.get(key, "")
            mask = getattr(row, cdr_cols[i])
            chains.append({"chain_name": chain_name, "mask": mask})

        region_map = _parse_regions(chains, max_length=max_len)

        # sequence
        seq = ["X"]  # BOS
        for i, col in enumerate(chain_cols):
            seq += list(getattr(row, col))
            if i < len(chain_cols) - 1:
                seq.append("X")  # separator
        seq.append("X")  # EOS
        seq += ["X"] * (max_len - len(seq))  # padding

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
                            "amino_acid": seq[pos],
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


def run_routing_analysis(model_name: str, model_path: str, config: RoutingConfig):

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, task="mlm")
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
    extracted = _process_outputs(data, config)
    extracted["model"] = model_name

    # save results
    data["balmmoe_output"] = data["balmmoe_output"].apply(_tensor_to_python)
    data.to_parquet(f"{config.output_dir}/results/{model_name}_raw-outputs.parquet")
    extracted.to_parquet(
        f"{config.output_dir}/results/{model_name}_routing_results.parquet"
    )
