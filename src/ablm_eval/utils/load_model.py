from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from balm import *  # needed to load custom balm models & tokenizer

from .esm_variants import checkpoint_keys, esm_variant_for_checkpoint

__all__ = ["load_model_and_tokenizer"]

# a classification model built from an MLM checkpoint has a new head and no MLM head
_CLS_NEW_PREFIXES = ("classifier.",)
_CLS_DROPPED_PREFIXES = ("lm_head.",)


def load_model_and_tokenizer(model_path: str, tokenizer_path: str, task: str, **kwargs):
    """Load a pretrained model and tokenizer.

    Gated-FFN ESM checkpoints (ablm-sweeps) are detected from their tensor names and built
    with the matching architecture. A checkpoint tensor the model has no slot for, or a model
    weight the checkpoint does not supply, raises rather than loading a different model.

    Args:
        model_path (str): Path to the pretrained model.
        task (str): The task, which determines the model type to load.
        kwargs: Any arguments passed here will be passed to the models
            `from_pretrained` function during loading.

    Raises:
        ValueError: If task is not "mlm" or "classification", or if the
            checkpoint's weights do not match the model built from it.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """

    tok_path = tokenizer_path if tokenizer_path is not None else model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    if task == "mlm":
        auto_cls = AutoModelForMaskedLM
    elif task == "classification":
        auto_cls = AutoModelForSequenceClassification
    else:
        raise ValueError(f"Unsupported task: {task}")

    config = AutoConfig.from_pretrained(model_path)
    with esm_variant_for_checkpoint(config, checkpoint_keys(model_path)):
        model, info = auto_cls.from_pretrained(model_path, output_loading_info=True, **kwargs)
    _check_loading_info(model, info, model_path, task)

    return model, tokenizer


def _check_loading_info(model, info: dict, model_path: str, task: str) -> None:
    """Raise if `from_pretrained` dropped, skipped or resized a weight it should have loaded."""
    missing = set(info.get("missing_keys", ())) - _tied_keys(model, info)
    unexpected = set(info.get("unexpected_keys", ()))
    mismatched = [k[0] for k in info.get("mismatched_keys", ())]
    if task == "classification":
        missing = {k for k in missing if not k.startswith(_CLS_NEW_PREFIXES)}
        unexpected = {k for k in unexpected if not k.startswith(_CLS_DROPPED_PREFIXES)}
    if missing or unexpected or mismatched:
        raise ValueError(
            f"{model_path}: checkpoint does not match the {type(model).__name__} built "
            f"from it. missing={sorted(missing)[:4]} ({len(missing)}), "
            f"unexpected={sorted(unexpected)[:4]} ({len(unexpected)}), "
            f"mismatched={sorted(mismatched)[:4]} ({len(mismatched)}). "
            "from_pretrained would drop or randomly initialize these weights."
        )


def _tied_keys(model, info: dict) -> set[str]:
    """Missing keys that share storage with a loaded weight (safetensors dedupes them)."""
    missing = set(info.get("missing_keys", ()))
    params = dict(model.named_parameters(remove_duplicate=False))
    loaded = {p.data_ptr() for k, p in params.items() if k not in missing}
    return {k for k in missing if k in params and params[k].data_ptr() in loaded}
