import os
from datetime import date

import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from ..utils import (
    load_model_and_tokenizer,
    load_and_tokenize,
    ComputeMetricsForSequenceClassification,
)
from ..configs import ClassificationConfig

__all__ = ["run_classification"]


def _def_training_args(run_name, config: ClassificationConfig):
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
        output_dir=f"{config.output_dir}/{config.model_name}/checkpoints/{run_name}",
        report_to=config.report_to,
        logging_dir=f"{config.output_dir}/{config.model_name}/logs/{run_name}",
    )
    return training_args


def run_classification(model_name: str, model_path: str, config: ClassificationConfig):

    # wandb
    if config.report_to == "wandb":
        for var_name in ["WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_JOB_TYPE"]:
            try:
                value = getattr(config, var_name.lower())
                if value is not None:
                    os.environ[var_name] = value
            except AttributeError:
                pass

    results = []
    for i in range(config.num_folds):

        # run name
        if config.run_name is None:
            run_name = f"{model_name}_{config.classification_name}_itr{i}_{date.today().isoformat()}"
        else:
            run_name = f"{config.run_name}_itr{i}_{date.today().isoformat()}"

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
            "train": f"{config.dataset_dir}/{config.file_prefix}_train{i}.csv",
            "test": f"{config.dataset_dir}/{config.file_prefix}_test{i}.csv",
        }

        # load & process dataset
        tokenized_dataset = load_and_tokenize(
            data_path=datasets, tokenizer=tokenizer, config=config
        )

        # inference
        trainer = Trainer(
            model,
            args=_def_training_args(run_name, config),
            tokenizer=tokenizer,
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
        metrics["itr"] = i
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"{config.output_dir}/results/{model_name}_classification.csv", index=False
    )
