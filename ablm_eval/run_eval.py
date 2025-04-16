import sys
from pathlib import Path

from .tasks.inference import run_inference

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
        type=pathlib.Path,
    )
    parser.add_argument(
        "--data_path",
        default="/home/jovyan/shared/Sarah/current/mixed-data_fx/data/large-scale/TTE/annotated/paired-sep-test-annotated_20241119.parquet",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--output_file",
        default="./results/inference-results.csv",
        type=str,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser()
    run_inference(args)