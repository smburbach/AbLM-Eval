import os
import json
import subprocess
import tempfile
from pathlib import Path

import yaml
import torch
import pandas as pd

from .classification_config import ClassificationConfig

__all__ = ["run_classification"]


def _set_wandb_vars(config: ClassificationConfig):
    for var_name in ["WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_JOB_TYPE"]:
        try:
            value = getattr(config, var_name.lower())
            if value is not None:
                os.environ[var_name] = value
        except AttributeError:
            pass


def _merge_results(
    temp_dir: str,
    results_file: str,
    dataset_name: str
):
    files = list(Path(temp_dir).glob("*.parquet"))
    merged_df = pd.concat([pd.read_parquet(f) for f in files])
    merged_df['dataset'] = dataset_name
    merged_df = merged_df.sort_values("itr")
    merged_df.to_csv(results_file, index=False)


def run_classification(model_name: str, model_path: str, config: ClassificationConfig):
    # wandb
    if config.report_to == "wandb":
        _set_wandb_vars(config)

    # convert config (data class) to json string
    config_json = json.dumps({key: value for key, value in config.__dict__.items()})

    # get training script path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "train_script.py")
    config_path = (
        f"{config.output_dir}/{config.dataset_name}_accelerate_config.yaml"
    )
    with open(config_path, "r") as f:
        accelerate_config = yaml.safe_load(f)

    # Determine the number of available GPUs
    total_gpus = torch.cuda.device_count()
    gpus_per_run = accelerate_config.get("num_processes")
    max_parallel_runs = total_gpus // gpus_per_run

    # Launch each GPU for its specific fold range
    fold_indices = list(range(config.num_folds))
    gpu_pool = list(range(total_gpus))
    processes = []
    with tempfile.TemporaryDirectory() as temp_dir:
        while fold_indices:
            for _ in range(min(max_parallel_runs, len(fold_indices))):
                fold = fold_indices.pop(0)
                port = 29500 + fold

                # gpus for this run
                assigned_gpus, gpu_pool = (
                    gpu_pool[:gpus_per_run],
                    gpu_pool[gpus_per_run:],
                )
                visible_devices = ",".join(str(gpu) for gpu in assigned_gpus)

                command = [
                    "accelerate", "launch",
                    "--main_process_port", str(port),
                    "--config_file", config_path,
                    script_path,
                    "--fold_itr", str(fold),
                    "--temp_dir", str(temp_dir),
                    "--config", config_json,
                    "--model_name", model_name,
                    "--model_path", model_path,
                ]
                p = subprocess.Popen(
                    command, env={**os.environ, "CUDA_VISIBLE_DEVICES": visible_devices}
                )
                processes.append((fold, p, assigned_gpus))

            # wait for this batch to finish running
            for fold, proc, released_gpus in processes:
                ret = proc.wait()
                if ret != 0:
                    print(f"Fold {fold} exited with error code {ret}")
                gpu_pool.extend(released_gpus)
            processes.clear()

        results_file = f"{config.output_dir}/results/{model_name}_{config.dataset_name}-classification.csv"
        _merge_results(temp_dir=temp_dir, results_file=results_file, dataset_name=config.dataset_name)
