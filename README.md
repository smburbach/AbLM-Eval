# AbLM-Eval

AbLM-Eval is a Python package designed for evaluating antibody language models (AbLMs) trained with an MLM strategy.
Tasks for evaluation include inference, classification, mutation prediction, naturalness prediction, and routing analysis for BALM-MoE models.
This repository provides a flexible framework for running these tasks and comparing results across multiple models.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/smburbach/AbLM-Eval.git
   cd AbLM-Eval
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Below is a guide on how to use the main functions in this repo.
You can also reference `example.py`.

### 1. Define Models

Specify the models you want to evaluate as a dictionary, where the keys are model names and the values are paths to the model directories:
```python
models = {
    "model-1": "/path/to/model_1/",
    "model-2": "/path/to/model_2/",
}
```

### 2. Define Task Configs

AbLM-Eval provides several configuration classes for different evaluation tasks. 
See individual task directories for more info about the available config parameters.

| Task | Config Name | Description |
| ---- | ----------- | ----------- |
| [Inference](ablm_eval/tasks/inference/) | `InferenceConfig` | MLM inference on single dataset |
| [Per-position Inference](ablm_eval/tasks/per_position_inference/) | `PerPositionConfig` | Per-position inference |
| [Mutation Prediction](ablm_eval/tasks/mutation_prediction/) | `MutationPredConfig` | Runs per-position inference and analysis accuracy of prediction mutations |
| [Naturalness Prediction](ablm_eval/tasks/naturalness_prediction/) | `NaturalnessConfig` | Runs per-position inference and calculates pseudo-perplexity & naturalness scores |
| [Classification](ablm_eval/tasks/classification/) | `ClassificationConfig` | Runs multi-fold classification and summarizes results across folds. |
| [Routing Analysis](ablm_eval/tasks/moe_routing/) | `RoutingConfig` | Analyzes routing of BALM-MoE models |


Create a list of the task configurations to run, for example:
```python
from ablm_eval import InferenceConfig, PerPositionConfig

configs = [
    InferenceConfig(
        data_path="/path/to/inference_data.parquet",
        batch_size=128,
        heavy_column="sequence_aa_heavy",
        light_column="sequence_aa_light",
    ),
    PerPositionConfig(
        data_path="/path/to/per_pos_data.parquet",
        heavy_column="sequence_aa_heavy",
        light_column="sequence_aa_light",
    ),
    # Add other configurations as needed
]
```

### 3. Define Output Directory
```python
shared_output_dir = "./results"
```
The results for each task will be saved in the `shared_output_dir` directory, with subdirectories for each task. Each task directory will contain:
- `results/`: Processed results for the task
- `config.json`: The configuration used for the task
- Comparison files (when `generate_comparisons=True` is set)

### 4. Run Tasks

Use the `evaluate_ablms` function to evaluate the models on the specified tasks:
```python
from ablm_eval import evaluate_ablms

evaluate_ablms(
    models=models,
    configs=configs,
    shared_output_dir=shared_output_dir,
    generate_comparisons=True, # creates tables & plots to compare model performance
    ignore_existing_files=False, # eval will fail if data already exists in the provided task directories
)
```

### 5. (Optional) Compare Results

If you have already run the evaluation and want to regenerate model comparisons, you can use the `compare_results` or `compare_task` functions.

####  `compare_results` : generates comparisons for all tasks in the provided directory

*Parameters:*
- `output_dir`: Directory containing all task results (should be the same as `shared_output_dir` used in `evaluate_ablms`)

```python
from ablm_eval import compare_results

compare_results(
    output_dir=shared_output_dir
)
```

#### `compare_task` : generates comparisons for for a single specified task

*Parameters:*
- `task_type`: Type of task to compare (e.g., "inference", "classification", "mutation_prediction", "routing", "naturalness")
- `task_results_dir`: Directory containing the results for this specific task
- `output_dir`: Directory where comparison outputs should be saved


```python
from ablm_eval import compare_task

compare_task(
    task_type="mutation_prediction",
    task_results_dir=f"./{shared_output_dir}/inference/results/",
    output_dir=f"./{shared_output_dir}/inference/"
)
```

