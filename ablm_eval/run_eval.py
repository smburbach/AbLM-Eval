import sys
import argparse
import pathlib

from .tasks.inference import run_inference
from .tasks import (
    run_inference,
    run_per_pos,
)

# defaults set for testing only
def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="test_model",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="/home/jovyan/shared/Sarah/current/BALM-MoE/training/checkpoints/BALM-MoE_500M_500k-stp_top2-8experts-capacity1-altsparsity_lr1e-4_bs32-8xGPUs_gelu_2025-04-11/checkpoint-300000/",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        type=str,
    )

    # task specific data
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=["inference", "per_pos_inference"],
    )
    parser.add_argument(
        "--inference_data", 
        type=str,
        default="/home/jovyan/shared/Sarah/current/mixed-data_fx/data/large-scale/TTE/annotated/paired-sep-test-annotated_20241119.parquet",
    )
    parser.add_argument(
        "--per_pos_data", 
        type=str,
        default= "/home/jovyan/shared/Sarah/current/mixed-data_fx/data/large-scale/TTE/annotated/paired-sep-1k-annotated.csv",
    )
    parser.add_argument(
        "--classification_kfold_dir", 
        type=str,
        default="/home/jovyan/shared/Sarah/current/curr-pMLM/eval/specificity-classification/data/TTE-5_HC/"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser()

    # build outputs dir: output_dir/model_name/
    output_path = pathlib.Path(args.output_dir) / args.model_name
    if output_path.exists() and any(output_path.iterdir()):
        if args.model_name != "test_model": # exception for testing
            raise Exception(
                f"The folder '{output_path}' already exists and is not empty! Please provide a different output directory or model name."
            )

    # tasks
    for task in args.task:
        if task == "inference":
            if not args.inference_data:
                raise ValueError("Missing --inference_data for inference task.")
            run_inference(args)
        elif task == "per_pos_inference":
            if not args.per_pos_data:
                raise ValueError("Missing --per_pos_data for per_pos task.")
            run_per_pos(args)
        elif task == "classification":
            if not args.classification_kfold_dir:
                raise ValueError("Missing --classification_kfold_dir for classification task.")
            run_per_pos(args)
        else:
            raise ValueError(f"Unknown task: {task}")