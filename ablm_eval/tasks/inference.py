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
    files = {
        "test": args.data_path
    }
    tokenized_dataset = load_and_tokenize(
        data_files=files,
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
            output_dir="./results/", report_to="none", per_device_eval_batch_size=64
        ),
    )
    res = trainer.evaluate(tokenized_dataset["test"])
    res["model"] = args.model_name
    res["model_path"] = str(args.model_path)

    res_df = pd.DataFrame([res])
    res_df.to_csv(args.output_file, index=False)
