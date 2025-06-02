# Inference Task

This module provides functionality for performing MLM inference on antibody-related datasets using the Hugging Face Trainer. 

## Usage

### 1. Define the configuration

Create a configuration using the `InferenceConfig` class. Refer to the [class docstring](inference_config.py) for detailed parameter descriptions.

Example:
```python
from ablm_eval import InferenceConfig

config = InferenceConfig(
    data_path="/path/to/dataset.parquet",
    batch_size=128,
    heavy_column="sequence_aa_heavy",
    light_column="sequence_aa_light",
)
```

### 2. Run task with `evaluate_ablms`

Using the `run_inference` command directly may result in unexpected behavior if the output directory is not setup correctly.

It is recommended to use the `evaluate_ablms` function to run the tasks, as it ensures that output directories are created correctly and results are organized consistently. See [here](../../../README.md) for more details.

### 3. Results
Results will be saved in the output directory ('results') like so:
```
results/inference/
|-- results/                         # Results for each model
|-- config.json                      # Task config file
|-- combined-inference-results.csv   # Combined results
```
