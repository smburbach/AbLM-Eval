import pathlib

import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from ..utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
    ComputeMetricsForMaskedLM,
)
from ..config import InferenceConfig

__all__ = ["run_inference"]

def run_inference(config: InferenceConfig):

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model_path, task="mlm")

    # load & process datatset
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
        compute_metrics=ComputeMetricsForMaskedLM(return_moe_losses=config.return_moe_losses),
        args=TrainingArguments(
            output_dir=config.output_dir, 
            report_to=config.report_to, 
            per_device_eval_batch_size=config.batch_size
        ),
    )
    results = trainer.evaluate(tokenized_dataset)
    results["model"] = config.model_name
    results["model_path"] = config.model_path

    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{config.output_dir}/{config.model_name}/test-inference.csv", index=False)
