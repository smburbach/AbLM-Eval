from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

__all__ = ["load_model_and_tokenizer"]


def load_model_and_tokenizer(model_path: str, task: str, **kwargs):
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

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if task == "mlm":
        model = AutoModelForMaskedLM.from_pretrained(model_path, **kwargs)
    elif task == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return model, tokenizer
