import argparse
import pathlib

import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from ..utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
    ComputeMetricsForMaskedLM,
)

__all__ = ["run_inference"]

def run_inference(args: argparse.Namespace):

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, task="mlm")

    # load & process datatset
    tokenized_dataset = load_and_tokenize(
        data_path=args.inference_data,
        tokenizer=tokenizer,
        heavy_column="sequence_aa_heavy",
        light_column="sequence_aa_light",
    )

    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # inference
    trainer = Trainer(
        model=model,
        data_collator=collator,
        compute_metrics=ComputeMetricsForMaskedLM(return_moe_losses=True),
        args=TrainingArguments(
            output_dir=args.output_dir, report_to="none", per_device_eval_batch_size=64
        ),
    )
    results = trainer.evaluate(tokenized_dataset)
    results["model"] = args.model_name
    results["model_path"] = str(args.model_path)

    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{args.output_dir}/{args.model_name}/test-inference.csv", index=False)
