from ablm_eval import (
    InferenceConfig,
    PerPositionConfig,
    ClassificationConfig,
    MutationPredConfig,
    RoutingConfig,
    run_eval,
)


def main():
    # model & output params
    shared_params = {
        "model_name": "test_model",
        "model_path": "/home/jovyan/shared/Sarah/current/BALM-MoE/training/checkpoints/BALM-MoE_500M_500k-stp_top2-8experts-capacity1-altsparsity_lr1e-4_bs32-8xGPUs_gelu_2025-04-11/checkpoint-300000/",
        "output_dir": "./results",
    }

    # define configs
    dir = "/home/jovyan/shared/Sarah/current/mixed-data_fx/data/large-scale/TTE/annotated/"
    configs = [
        InferenceConfig(
            inference_data=f"{dir}paired-sep-test-annotated_20241119.parquet",
            batch_size=128,
            heavy_column="sequence_aa_heavy",
            light_column="sequence_aa_light",
            **shared_params,
        ),
        PerPositionConfig(
            per_pos_data=f"{dir}paired-sep-1k-annotated.csv",
            **shared_params,
        ),
        ClassificationConfig(
            dataset_dir="/home/jovyan/shared/Sarah/current/curr-pMLM/eval/specificity-classification/data/TTE-5_HFC/",
            file_prefix="hd-0_flu-1_cov-2",
            classification_name="HD-Flu-CoV",
            heavy_column="h_sequence",
            light_column="l_sequence",
            num_folds=1,
            num_classes=3,
            epochs=1,
            eval_steps=10,
            report_to="none",
            **shared_params,
        ),
    ]

    run_eval(configs)


if __name__ == "__main__":
    main()
