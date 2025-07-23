# Per-position Inference Task

This module provides functionality to run per-position inference, where each position in a sequence is individually masked and predicted by the model.

## Usage

### 1. Define the configuration

Create a configuration using the `PerPositionConfig` class. Refer to the [class docstring](per_position_inference_config.py) for detailed parameter descriptions.

Example:
```python
from ablm_eval import PerPositionConfig

config = PerPositionConfig(
    antibody_datatype="unpaired",
    data_path="/path/to/per_pos_data.parquet",
    dataset_columns=DatasetColumns(
        id_column="seq_id", chain_columns=["sequence"]
    ),
)
```

### 2. Run task with `evaluate_ablms`

Using the `run_per_pos` command directly may result in unexpected behavior if the output directory is not setup correctly.

It is recommended to use the `evaluate_ablms` function to run the tasks, as it ensures that output directories are created correctly and results are organized consistently. See [here](../../../README.md) for more details.

### 3. Results
Results will be saved in the output directory ('results') like so:
```
results/per_pos_inference/
|-- results/                                              # Results for each model
    |-- {model-name}_per-position-inference.parquet
|-- config.json                                           # Task config file
|-- combined-results_mutated_accuracy.png
|-- combined-results_mutated_median_loss.png
|-- combined-results_unmutated_accuracy.png
|-- combined-results_unmutated_median_loss.png
|-- results-summary_per-pos_inference.csv
```
