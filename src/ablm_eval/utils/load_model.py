from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from balm import *  # needed to load custom balm models & tokenizer

__all__ = ["load_model_and_tokenizer"]


def load_model_and_tokenizer(model_path: str, tokenizer_path: str, task: str, **kwargs):
    """Load a pretrained model and tokenizer.

    Args:
        model_path (str): Path to the pretrained model.
        task (str): The task, which determines the model type to load.
        kwargs: Any arguments passed here will be passed to the models
            `from_pretrained` function during loading.

    Raises:
        ValueError: If task is not "mlm" or "classification".

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """

    tok_path = tokenizer_path if tokenizer_path is not None else model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    if task == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(model_path, **kwargs)
    elif task == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return model, tokenizer
