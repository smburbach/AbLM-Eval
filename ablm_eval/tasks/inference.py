import pathlib

import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from ..utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
    ComputeMetricsForMaskedLM,
)
from ..configs import InferenceConfig

__all__ = ["run_inference"]


def run_inference(model_name: str, model_path: str, config: InferenceConfig):

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, task="mlm")

    # load & process dataset
    tokenized_dataset = load_and_tokenize(
        data_path=config.inference_data,
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

    # save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{config.output_dir}/results/{model_name}_inference.csv", index=False)
