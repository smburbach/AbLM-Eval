import json
import argparse
from datetime import date

import pandas as pd
from transformers import TrainingArguments, Trainer

from ablm_eval.utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
    ComputeMetricsForSequenceClassification,
)
from ablm_eval.tasks import ClassificationConfig
from ablm_eval.utils import DatasetColumns


def _parse_config(config_json):
    config_dict = json.loads(config_json)

    # convert dataset_columns dict to DatasetColumns object
    if isinstance(config_dict.get("dataset_columns"), dict):
        config_dict["dataset_columns"] = DatasetColumns(
            **config_dict["dataset_columns"]
        )

    # convert training_args dict to TrainingArguments object
    if isinstance(config_dict.get("training_args"), dict):
        config_dict["training_args"] = TrainingArguments(**config_dict["training_args"])

    # convert data_path keys to ints
    if "data_path" in config_dict:
        config_dict["data_path"] = {
            int(k): v for k, v in config_dict["data_path"].items()
        }

    return ClassificationConfig(**config_dict)


def _def_training_args(run_name, config):
    if hasattr(config, "training_args") and config.training_args is not None:
        training_args = config.training_args
    else:
        # fallback to defaults
        training_args = TrainingArguments(
            run_name=run_name,
            seed=42,
            bf16=True,
            learning_rate=1e-4,
            per_device_train_batch_size=32,
            num_train_epochs=3,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            eval_strategy="steps",
            eval_steps=250,
            per_device_eval_batch_size=128,
            eval_accumulation_steps=50,
            logging_steps=50,
            save_strategy="no",
            logging_first_step=True,
            output_dir=f"{config.output_dir}/checkpoints/{run_name}",
            logging_dir=f"{config.output_dir}/logs/{run_name}",
            report_to="none",
        )

    # always enforce output_dir and run_name
    training_args.output_dir = f"{config.output_dir}/checkpoints/{run_name}"
    training_args.run_name = run_name

    return training_args


def main(
    model_name: str,
    model_path: str,
    fold_itr: str,
    temp_dir: str,
    config: ClassificationConfig,
):

    # run name
    if config.run_name is None:
        run_name = f"{model_name}_{config.dataset_name}_itr{fold_itr}_{date.today().isoformat()}"
    else:
        run_name = f"{config.run_name}_itr{fold_itr}_{date.today().isoformat()}"

    # load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        tokenizer_path=config.tokenizer_path,
        task="classification",
        num_labels=config.num_classes,
        attention_classifier=config.attention_classifier,
    )
    if config.manually_freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False

    datasets = config.data_path[int(fold_itr)]

    # load & process dataset
    tokenized_dataset = load_and_tokenize(
        data_path=datasets, tokenizer=tokenizer, config=config
    )

    # inference
    trainer = Trainer(
        model,
        args=_def_training_args(run_name, config),
        processing_class=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=ComputeMetricsForSequenceClassification(
            positive_label=config.positive_label,
            num_classes=config.num_classes,
            multi_class_average=config.multi_class_average,
        ),
    )
    trainer.train()

    # final eval
    _, _, metrics = trainer.predict(tokenized_dataset["test"])
    metrics["model"] = model_name
    metrics["model_path"] = model_path
    metrics["itr"] = fold_itr

    # save
    results_df = pd.DataFrame([metrics])
    results_df.to_parquet(f"{temp_dir}/{model_name}_itr{fold_itr}.parquet")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--temp_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold_itr", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    config = _parse_config(args.config)

    main(
        model_name=args.model_name,
        model_path=args.model_path,
        fold_itr=args.fold_itr,
        temp_dir=args.temp_dir,
        config=config,
    )
