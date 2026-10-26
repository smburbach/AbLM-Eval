"""Gated-FFN ESM checkpoints that stock HuggingFace ESM cannot load.

ablm-sweeps (``esm2/12_sota_convergence/training_mods/model.py``) trains gated-FFN models by
swapping a subclass into ``transformers.models.esm.modeling_esm`` before building a stock
``EsmForMaskedLM``. The checkpoint then carries an extra ``intermediate.up`` projection per
layer, and ``config.hidden_act`` holds the gate activation (``silu`` for SwiGLU). Without the
same swap, ``from_pretrained`` drops the ``up`` tensors and every layer runs
``gelu(dense(x))``: ~uniform predictions, no exception.
"""

import json
import os
from contextlib import contextmanager

import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.models.esm import modeling_esm

__all__ = ["checkpoint_keys", "esm_variant_for_checkpoint"]

_GATED_SUFFIX = ".intermediate.up.weight"


class EsmIntermediateGLU(modeling_esm.EsmIntermediate):
    """3-matrix gated FFN: ``act(W_gate x) * (W_up x)``.

    ``W_gate`` is the stock ``dense`` and ``EsmOutput`` stays the down-projection, so the only
    added parameter is ``up``. The checkpoint's ``intermediate_size`` is already the value the
    trainer built with, so nothing is rescaled.
    """

    def __init__(self, config):
        super().__init__(config)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.act_fn(self.dense(hidden_states)) * self.up(hidden_states)


def checkpoint_keys(model_path: str) -> set[str]:
    """Tensor names in a checkpoint's safetensors files, read from the headers alone.

    Args:
        model_path (str): Local checkpoint directory or HuggingFace Hub repo id.

    Returns:
        set[str]: The tensor names, or an empty set if no safetensors weights were found.
    """
    if os.path.isdir(model_path):
        index = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index) as f:
                return set(json.load(f)["weight_map"])
        single = os.path.join(model_path, "model.safetensors")
        if not os.path.isfile(single):
            return set()
        with open(single, "rb") as f:
            length = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(length))
        return {k for k in header if k != "__metadata__"}

    from huggingface_hub import get_safetensors_metadata

    try:
        return set(get_safetensors_metadata(model_path).weight_map)
    except Exception:
        return set()


@contextmanager
def esm_variant_for_checkpoint(config, keys: set[str]):
    """Build ESM models with the checkpoint's gated FFN for the duration of the block.

    A no-op for non-ESM configs and for ESM checkpoints without ``intermediate.up`` tensors.
    ESM resolves ``EsmIntermediate`` from module globals at construction, so the swap applies
    to every ESM head class and is undone on exit.

    Args:
        config: The checkpoint's ``PretrainedConfig``.
        keys (set[str]): The checkpoint's tensor names (see :func:`checkpoint_keys`).
    """
    gated = getattr(config, "model_type", None) == "esm" and any(
        k.endswith(_GATED_SUFFIX) for k in keys
    )
    if not gated:
        yield
        return
    original = modeling_esm.EsmIntermediate
    modeling_esm.EsmIntermediate = EsmIntermediateGLU
    try:
        yield
    finally:
        modeling_esm.EsmIntermediate = original
