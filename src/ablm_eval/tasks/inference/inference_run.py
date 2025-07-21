import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from ...utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
    ComputeMetricsForMaskedLM,
)
from .inference_config import InferenceConfig

__all__ = ["run_inference"]


def run_inference(model_name: str, model_path: str, config: InferenceConfig):

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, task="mlm")
    model.eval()

    # load & process dataset
    tokenized_dataset = load_and_tokenize(
        data_path=config.data_path,
        tokenizer=tokenizer,
        config=config,
    )

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=config.mlm,
        mlm_probability=config.mlm_probability,
    )

    # inference
    trainer = Trainer(
        model=model,
        data_collator=collator,
        compute_metrics=ComputeMetricsForMaskedLM(
            return_moe_losses=config.return_moe_losses
        ),
        args=TrainingArguments(
            output_dir=config.output_dir,
            report_to=config.report_to,
            per_device_eval_batch_size=config.batch_size,
        ),
    )
    results = trainer.evaluate(tokenized_dataset)
    results["model"] = model_name
    results["model_path"] = model_path
    results["dataset"] = config.dataset_name

    # save results
    results_df = pd.DataFrame([results])
    data_name = f"{config.dataset_name}-" if config.dataset_name is not None else ""
    results_df.to_csv(
        f"{config.output_dir}/results/{model_name}_{data_name}inference.csv",
        index=False,
    )
