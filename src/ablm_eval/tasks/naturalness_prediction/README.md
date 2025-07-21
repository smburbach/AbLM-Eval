# Naturalness Analysis Task

This module provides functionality to run [per-position inference](../per_position_inference/) on sequences and calculate the predicted pseudo-perplexity and 'naturalness' of that sequence.

## Usage

### 1. Define the configuration

Create a configuration using the `NaturalnessConfig` class. Refer to the [class docstring](naturalness_config.py) for detailed parameter descriptions.

Example:
```python
from ablm_eval import NaturalnessConfig

config = NaturalnessConfig(
    dataset_name="test",
    data_path=f"/path/to/dataset.csv",
    sequence_column="sequence_aa",
)
```

### 2. Run task with `evaluate_ablms`

Using the `run_naturalness` command directly may result in unexpected behavior if the output directory is not setup correctly.

It is recommended to use the `evaluate_ablms` function to run the tasks, as it ensures that output directories are created correctly and results are organized consistently. See [here](../../../README.md) for more details.

### 3. Results
Results will be saved in the output directory ('results') like so:
```
results/naturalness/
|-- results/                                              # Results for each model
    |-- {model-name}_per-position-inference.parquet       # Per-position inference results
    |-- {model-name}_naturalness.parquet                  # Processed naturalness results
|-- config.json                                           # Task config file
|-- *naturalness.png                                      # Plot(s) of predicted naturalness between datasets and/or models
```
