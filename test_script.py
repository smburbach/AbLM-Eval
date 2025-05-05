from ablm_eval import (
    InferenceConfig,
    PerPositionConfig,
    ClassificationConfig,
    MutationPredConfig,
    RoutingConfig,
    evaluate_ablms,
    compare_results,
    compare_task
)


def main():
    # models
    models_dir = "/home/jovyan/shared/Sarah/current/BALM-MoE/moe-optimization/"
    models = {
        "top2": f"{models_dir}top2_capacity/models/BALM-MoE_45M-act_top2_capacity1.0_8experts-1shared-fixed_lr1e-4_bs32-4xGPUs_swiglu_2025-04-25/",
        "expert-choice": f"{models_dir}expert-choice_capacity/models/BALM-MoE_45M-act_expertchoice_capacity0.5_8experts-1shared-fixed_lr1e-4_bs32-4xGPUs_swiglu_2025-04-25/",
    }
    shared_output_dir = "./results"

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
        #     data_path=f"{dir}paired-sep-1k-annotated.csv",
        #     heavy_column="sequence_aa_heavy",
        #     light_column="sequence_aa_light",
        # ),
        MutationPredConfig(
            data_path=f"{dir}paired-sep-1k-annotated.csv",
            sequence_column="sequence_mutated" # "sequence_mutated" or "sequence_germ"
        ),
        # ClassificationConfig(
        #     dataset_dir="/home/jovyan/shared/Sarah/current/curr-pMLM/eval/specificity-classification/data/TTE-5_HC/",
        #     file_prefix="hd-0_cov-1",
        #     classification_name="HD-CoV",
        #     heavy_column="h_sequence",
        #     light_column="l_sequence",
        #     num_folds=5,
        #     num_classes=2,
        #     epochs=1,
        #     eval_steps=300,
        #     report_to="none",
        # ),
        # RoutingConfig(
        #     routing_data=f"{dir}paired-sep-1k-annotated.csv",
        #     heavy_column="sequence_aa_heavy",
        #     light_column="sequence_aa_light",
        # )
    ]

    evaluate_ablms(
        models, 
        configs, 
        shared_output_dir, 
        generate_comparisons=True,
        ignore_existing_files=True,
    )
    # compare_results(output_dir=shared_output_dir)
    # compare_task(
    #     task_type="mutation_prediction",
    #     task_results_dir="./results/mutation_prediction/results/",
    #     output_dir="./results/mutation_prediction/"
    # )


if __name__ == "__main__":
    main()
