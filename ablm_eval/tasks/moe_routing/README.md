# MoE Routing Task

This module provides functionality to analyze the MoE routing of [BALM-MoE](https://github.com/brineylab/BALM) models. This code is not compatible with other MoE implementations.

## Usage

### 1. Define the configuration

Create a configuration using the `InferenceConfig` class. Refer to the [class docstring](inference_config.py) for detailed parameter descriptions.

Example:
```python
from ablm_eval import RoutingConfig

config = RoutingConfig(
    data_path=f"/path/to/dataset.csv",
    heavy_column="sequence_aa_heavy",
    light_column="sequence_aa_light",
)
```

### 2. Run task with `evaluate_ablms`

Using the `run_routing_analysis` command directly may result in unexpected behavior if the output directory is not setup correctly.

It is recommended to use the `evaluate_ablms` function to run the tasks, as it ensures that output directories are created correctly and results are organized consistently. See [here](../../../README.md) for more details.

### 3. Results
Results will be saved in the output directory ('results') like so:
```
results/routing_analysis/
|-- results/                            # Results for each model
    |-- {model-name}_raw-outputs.parquet       # Raw model logits
    |-- {model-name}_routing_results.parquet   # Processed routing results
|-- config.json                         # Task config file
|-- routing_summary.csv                 # Routing summary per region
|-- CDRH3-compare.png                   # Plot comparing CDRH3 routing across models
|-- {model-name}_region-heatmaps.png    # Heatmaps of FR & CDR region routing for each model
```
