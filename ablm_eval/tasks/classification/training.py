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
from ablm_eval.configs import ClassificationConfig


def _parse_config(config_json):
    config_dict = json.loads(config_json)
    return ClassificationConfig(**config_dict)


def _def_training_args(run_name, model_name, config: ClassificationConfig):
    training_args = TrainingArguments(
        run_name=run_name,
        seed=config.seed,
        bf16=config.bf16,
        fp16=config.fp16,
        # train
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        num_train_epochs=config.epochs,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        # eval
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        per_device_eval_batch_size=config.eval_batch_size,
        eval_accumulation_steps=config.eval_accumulation_steps,
        # saving & logging
        logging_first_step=config.logging_first_step,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        output_dir=f"{config.output_dir}/checkpoints/{run_name}",
        report_to=config.report_to,
        logging_dir=f"{config.output_dir}/logs/{run_name}",
    )
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
        model_path,
        task="classification",
        num_labels=config.num_classes,
        attention_classifier=config.attention_classifier,
    )
    if config.manually_freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False

    datasets = {
        "train": f"{config.dataset_dir}/{config.file_prefix}_train{fold_itr}.csv",
        "test": f"{config.dataset_dir}/{config.file_prefix}_test{fold_itr}.csv",
    }

    # load & process dataset
    tokenized_dataset = load_and_tokenize(
        data_path=datasets, tokenizer=tokenizer, config=config
    )

    # inference
    trainer = Trainer(
        model,
        args=_def_training_args(run_name, model_name, config),
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
