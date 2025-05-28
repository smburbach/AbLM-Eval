import os

from balm import BalmConfig, BalmForMaskedLM, BalmMoEConfig, BalmMoEForMaskedLM
from transformers import EsmConfig, EsmForMaskedLM

__all__ = ["generate_mini_models"]


def generate_mini_models(output_dir):
    """
    Generate and save mini versions of BALM, BALM-MoE, and ESM models.
    Returns a dict mapping model names to their output paths.
    """
    paths = {}

    # BALM dense
    balm_config = BalmConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_size=16
    )
    balm_dense_path = os.path.join(output_dir, "mini-BALM-dense")
    BalmForMaskedLM(config=balm_config).save_pretrained(balm_dense_path)
    paths["BALM-dense"] = balm_dense_path

    # BALM MoE
    balm_moe_config = BalmMoEConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_size=16,
        num_experts=2
    )
    balm_moe_path = os.path.join(output_dir, "mini-BALM-MoE")
    BalmMoEForMaskedLM(config=balm_moe_config).save_pretrained(balm_moe_path)
    paths["BALM-MoE"] = balm_moe_path

    # ESM
    esm_config = EsmConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_size=16,
        vocab_size=32
    )
    esm_path = os.path.join(output_dir, "mini-ESM")
    EsmForMaskedLM(config=esm_config).save_pretrained(esm_path)
    paths["ESM"] = esm_path

    return paths
