import argparse
import pathlib

import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from ..utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
    ComputeMetricsForMaskedLM,
)


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default=None,
        required=True,
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default=None,
        required=True,
        type=pathlib.Path,
    )
    parser.add_argument(
        "--output_file",
        default="./inference-results.csv",
        type=str,
    )
    args = parser.parse_args()
    return args


def main():
    # read args
    args = parser()

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # load & process datatset
    files = {
        "test": "/home/jovyan/shared/Sarah/current/mixed-data_fx/data/large-scale/TTE/annotated/paired-sep-test-annotated_20241119.parquet"
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
        compute_metrics=ComputeMetricsForMaskedLM,
        args=TrainingArguments(
            output_dir="./results/", report_to="none", per_device_eval_batch_size=64
        ),
    )
    res = trainer.evaluate(tokenized_dataset["test"])
    res["model"] = res.model_name
    res["model_path"] = res.model_path

    res_df = pd.DataFrame(res)
    res_df.to_csv(args.output_file)


if __name__ == "__main__":
    main()
