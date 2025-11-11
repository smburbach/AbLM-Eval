import os
import json
import subprocess
import tempfile
from pathlib import Path
import itertools

import yaml
import torch
import pandas as pd
from transformers.training_args import TrainingArguments

from .classification_config import ClassificationConfig
from ...utils import DatasetColumns

__all__ = ["run_classification"]


def _set_wandb_vars(config: ClassificationConfig):
    for var_name in ["WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_JOB_TYPE"]:
        try:
            value = getattr(config, var_name.lower())
            if value is not None:
                os.environ[var_name] = value
        except AttributeError:
            pass


def _to_serializable_dict(config: ClassificationConfig):
    config_dict = config.__dict__.copy()
    # Convert TrainingArguments to dict if present
    if isinstance(config_dict.get("training_args"), TrainingArguments):
        config_dict["training_args"] = config.training_args.to_dict()
    # Convert DatasetColumns to dict if present
    if isinstance(config_dict.get("dataset_columns"), DatasetColumns):
        config_dict["dataset_columns"] = config.dataset_columns.to_dict()
    return config_dict


def _merge_results(temp_dir: str, results_file: str, dataset_name: str):
    files = list(Path(temp_dir).glob("*.parquet"))
    merged_df = pd.concat([pd.read_parquet(f) for f in files])
    merged_df["dataset"] = dataset_name
    merged_df = merged_df.sort_values("itr")
    merged_df.to_csv(results_file, index=False)


def run_classification(model_name: str, model_path: str, config: ClassificationConfig):
    # wandb
    _set_wandb_vars(config)

    # convert config (data class) to json string
    config_json = json.dumps(_to_serializable_dict(config))

    # get training script path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "train_script.py")
    config_path = f"{config.output_dir}/{config.dataset_name}_accelerate_config.yaml"
    with open(config_path, "r") as f:
        accelerate_config = yaml.safe_load(f)

    # determine available & required GPUs
    total_gpus = torch.cuda.device_count()
    gpus_per_run = accelerate_config.get("num_processes", 1)
    max_parallel_runs = max(1, total_gpus // gpus_per_run)

    # track GPUs
    fold_indices = list(range(config.num_folds))
    gpu_ids = list(range(total_gpus))
    gpu_cycle = itertools.cycle(gpu_ids)  # infinite iterator

    processes = []
    with tempfile.TemporaryDirectory() as temp_dir:
        while fold_indices:
            running = []
            for _ in range(min(max_parallel_runs, len(fold_indices))):
                fold = fold_indices.pop(0)
                port = 29500 + fold

                # assign GPUs for this run
                assigned_gpus = [next(gpu_cycle) for _ in range(gpus_per_run)]
                visible_devices = ",".join(str(gpu) for gpu in assigned_gpus)

                # shared args
                shared = [
                    f"--fold_itr={str(fold)}",
                    f"--temp_dir={str(temp_dir)}",
                    f"--config={config_json}",
                    f"--model_name={model_name}",
                    f"--model_path={model_path}",
                ]
                # launcher args
                if config.launcher == "accelerate":
                    launcher = [
                        "accelerate",
                        "launch",
                        f"--main_process_port={str(port)}",
                        f"--config_file={config_path}",
                        script_path,
                    ]
                elif config.launcher == "python":
                    launcher = ["python", script_path]
                else:
                    raise ValueError(
                        "Launcher must be either 'python' or 'accelerate'."
                    )
                # final command
                command = launcher + shared
                p = subprocess.Popen(
                    command, env={**os.environ, "CUDA_VISIBLE_DEVICES": visible_devices}
                )
                running.append((fold, p, assigned_gpus))

            # Wait for this batch to finish
            for fold, proc, _ in running:
                ret = proc.wait()
                if ret != 0:
                    print(f"Fold {fold} exited with error code {ret}")

        results_file = f"{config.output_dir}/results/{model_name}_{config.dataset_name}-classification.csv"
        _merge_results(
            temp_dir=temp_dir,
            results_file=results_file,
            dataset_name=config.dataset_name,
        )
