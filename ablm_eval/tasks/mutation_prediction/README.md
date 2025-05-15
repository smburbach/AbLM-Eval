# Mutation Prediction Task

This module provides functionality to run [per-position inference](../per_position_inference/) on sequences and analyze the predicted mutations for positional, chemical, and amino-acid matches.

## Usage

### 1. Define the configuration

Create a configuration using the `MutationPredConfig` class. Refer to the [class docstring](mutation_pred_config.py) for detailed parameter descriptions.

Example:
```python
from ablm_eval import MutationPredConfig

config = MutationPredConfig(
    data_path=f"/path/to/airr-formatted-dataset.parquet",
    sequence_column="sequence_germ",
)
```

### 2. Run task with `evaluate_ablms`

Using the `run_mutation_pred` command directly may result in unexpected behavior if the output directory is not setup correctly.

It is recommended to use the `evaluate_ablms` function to run the tasks, as it ensures that output directories are created correctly and results are organized consistently. See [here](../../../README.md) for more details.

### 3. Results
Results will be saved in the output directory ('results') like so:
```
results/mutation_prediction/
|-- results/                                              # Results for each model
    |-- {model-name}_per-position-inference.parquet       # Per-position inference results
    |-- {model-name}_mutation-analysis.parquet            # Processed mutation analysis results
|-- config.json                                           # Task config file
|-- all-predicted-mutations_histogram.png                 # Histogram of all predicted mutations
|-- true-positions-predicted-mutations_histogram.png      # Histogram of correctly positioned mutations
|-- mutation-match-ratios.png                             # Plot ratios of correctly predicted mutations
```
