# Classification Task

This module provides functionality for performing classification tasks on antibody-related datasets using the Hugging Face Trainer. It supports multi-class and binary classification, cross-validation, and integration with Weights & Biases (WandB) for experiment tracking.

## Usage

### 1. Define a Configuration

Create a configuration using the `ClassificationConfig` class. Refer to the [class docstring](classification_config.py) for detailed parameter descriptions.

Example:
```python
from ablm_eval import ClassificationConfig

config = ClassificationConfig(
    dataset_name="HD-Flu-CoV",
    data_path = {
        i: {
            "train": f"{class_dir}/hd-0_flu-1_cov-2_train{i}.csv",
            "test": f"{class_dir}/hd-0_flu-1_cov-2_test{i}.csv"
        }
        for i in range(5)
    },
    antibody_datatype="paired",
    launcher="accelerate",
    wandb_project="HD-Flu-CoV_classification",
    dataset_columns=DatasetColumns(chain_columns=["h_sequence", "l_sequence"]),
    num_folds=5,
    num_classes=3,
    manually_freeze_base=True,
    training_args=TrainingArguments(
        seed=42,
        fp16=True,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=250,
        per_device_eval_batch_size=32,
        eval_accumulation_steps=50,
        logging_steps=50,
        save_strategy="no",
        logging_first_step=True,
        report_to="wandb",
        log_level="debug",
    ),
)
```

### 2. Run task with `evaluate_ablms`

Using the `run_classification` command directly may result in unexpected behavior if the output directory is not setup correctly.

It is recommended to use the `evaluate_ablms` function to run the tasks, as it ensures that output directories are created correctly and results are organized consistently. See [here](../../../README.md) for more details.

### 3. Results
Results will be saved in the output directory ('results') like so:
```
results/HD-CoV_classification/
|-- checkpoints/                          # Checkpoints for each fold
|-- results/                              # Results for each model, with folds separated
|-- config.json                           # Task config file
|-- HD-CoV_accelerate_config.yaml         # Accelerate config file, if launcher="accelerate"
|-- combined-classification-results.csv   # Results averaged across folds
```
