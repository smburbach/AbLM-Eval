from contextlib import nullcontext

import pytest
import torch
from transformers.models.esm import modeling_esm

import ablm_eval.utils.load_model as load_model_module
from ablm_eval.utils import load_model_and_tokenizer
from ablm_eval.utils.esm_variants import EsmIntermediateGLU, checkpoint_keys

GATED = "sota_best"  # ablm-sweeps AR8 SwiGLU, 72 layers
STOCK_ESM = "sota_baseline"
BALM = "balm_paired"


def _masked_accuracy(model, tokenizer, sequences: list[str]) -> float:
    """Top-1 accuracy at 15% randomly masked non-special positions, fixed seed."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    batch = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = batch["input_ids"]
    special = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
            for ids in input_ids.tolist()
        ],
        dtype=torch.bool,
    )
    gen = torch.Generator().manual_seed(0)
    masked = (torch.rand(input_ids.shape, generator=gen) < 0.15) & ~special
    inputs = input_ids.masked_fill(masked, tokenizer.mask_token_id)
    with torch.no_grad():
        logits = model(
            input_ids=inputs.to(device), attention_mask=batch["attention_mask"].to(device)
        ).logits
    preds = logits.argmax(-1).cpu()
    return (preds[masked] == input_ids[masked]).float().mean().item()


def test_gated_checkpoint_builds_gated_ffn(models_dir):
    path = str(models_dir / GATED)
    model, _ = load_model_and_tokenizer(path, None, task="mlm")
    layers = model.esm.encoder.layer
    assert all(isinstance(layer.intermediate, EsmIntermediateGLU) for layer in layers)
    assert modeling_esm.EsmIntermediate is not EsmIntermediateGLU


def test_gated_checkpoint_predicts(models_dir, eval_sequences):
    model, tokenizer = load_model_and_tokenizer(str(models_dir / GATED), None, task="mlm")
    assert _masked_accuracy(model, tokenizer, eval_sequences) > 0.5


def test_gated_checkpoint_without_variant_raises(models_dir, monkeypatch):
    monkeypatch.setattr(
        load_model_module, "esm_variant_for_checkpoint", lambda *a, **k: nullcontext()
    )
    with pytest.raises(ValueError, match=r"unexpected=\['esm\.encoder\.layer\."):
        load_model_and_tokenizer(str(models_dir / GATED), None, task="mlm")


@pytest.mark.parametrize("name", [GATED, STOCK_ESM, BALM])
def test_real_checkpoints_load_strictly(models_dir, name):
    model, _ = load_model_and_tokenizer(str(models_dir / name), None, task="mlm")
    assert model is not None


def test_stock_esm_has_no_gate_and_tied_decoder(models_dir):
    path = str(models_dir / STOCK_ESM)
    assert not any(k.endswith(".intermediate.up.weight") for k in checkpoint_keys(path))
    model, _ = load_model_and_tokenizer(path, None, task="mlm")
    assert type(model.esm.encoder.layer[0].intermediate) is modeling_esm.EsmIntermediate
    assert (
        model.lm_head.decoder.weight.data_ptr()
        == model.esm.embeddings.word_embeddings.weight.data_ptr()
    )


@pytest.mark.parametrize("name", [GATED, STOCK_ESM])
def test_classification_from_mlm_checkpoint(models_dir, name):
    model, _ = load_model_and_tokenizer(
        str(models_dir / name), None, task="classification", num_labels=2
    )
    assert model.config.num_labels == 2
