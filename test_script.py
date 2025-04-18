from ablm_eval import (
    InferenceConfig,
    PerPositionConfig,
    ClassificationConfig,
    MutationPredConfig,
    RoutingConfig,
    eval_and_compare,
)


def main():
    # models
    models = {
        ("test_model_20k", "/home/jovyan/shared/Sarah/current/BALM-MoE/checkpoints/BALM-MoE_160M_50k-stp_1shared-top1-0.25capacity-8experts-altsparsity_lr1e-4_bs32-4xGPUs_2025-04-18/checkpoint-20000/"),
        ("test_model_50k", "/home/jovyan/shared/Sarah/current/BALM-MoE/checkpoints/BALM-MoE_160M_50k-stp_1shared-top1-0.25capacity-8experts-altsparsity_lr1e-4_bs32-4xGPUs_2025-04-18/checkpoint-50000/"),
    }
    shared_output_dir = "./topk_results"

    # define configs
    dir = "/home/jovyan/shared/Sarah/current/mixed-data_fx/data/large-scale/TTE/annotated/"
    configs = [
        # InferenceConfig(
        #     inference_data=f"{dir}paired-sep-test-annotated_20241119.parquet",
        #     batch_size=128,
        #     heavy_column="sequence_aa_heavy",
        #     light_column="sequence_aa_light",
        # ),
        # PerPositionConfig(
        #     per_pos_data=f"{dir}paired-sep-1k-annotated.csv",
        # ),
        # ClassificationConfig(
        #     dataset_dir="/home/jovyan/shared/Sarah/current/curr-pMLM/eval/specificity-classification/data/TTE-5_HFC/",
        #     file_prefix="hd-0_flu-1_cov-2",
        #     classification_name="HD-Flu-CoV",
        #     heavy_column="h_sequence",
        #     light_column="l_sequence",
        #     num_folds=1,
        #     num_classes=3,
        #     epochs=1,
        #     eval_steps=10,
        #     report_to="none",
        # ),
        RoutingConfig(
            routing_data=f"{dir}paired-sep-1k-annotated.csv",
        )
    ]

    eval_and_compare(models, configs, shared_output_dir)


if __name__ == "__main__":
    main()
