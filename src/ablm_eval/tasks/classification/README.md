# Classification Task

This module provides functionality for performing classification tasks on antibody-related datasets using the Hugging Face Trainer. It supports multi-class and binary classification, cross-validation, and integration with Weights & Biases (WandB) for experiment tracking.

## Usage

### 1. Define a Configuration

Create a configuration using the `ClassificationConfig` class. Refer to the [class docstring](classification_config.py) for detailed parameter descriptions.

Example:
```python
from ablm_eval import ClassificationConfig

config = ClassificationConfig(
    dataset_dir="/path/to/dataset",
    file_prefix="hd-0_cov-1", # for the file 'hd-0_cov-1_train{i}.csv' where i is the fold number
    dataset_name="HD-CoV",
    sequence_column="sequence",
    num_folds=5,
    num_classes=2,
    epochs=3,
    train_batch_size=32,
    report_to="wandb",
    wandb_project="HD-CoV_classification",
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
|-- HD-CoV_accelerate_config.yaml         # Accelerate config file
|-- combined-classification-results.csv   # Results averaged across folds
```
