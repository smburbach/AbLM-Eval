from ablm_eval import (
    InferenceConfig,
    PerPositionConfig,
    MutationPredConfig,
    ClassificationConfig,
    RoutingConfig,
    NaturalnessConfig,
    evaluate_ablms,
    compare_results,
    compare_task,
)


def main():
    # models
    models = {
        "model-1": "/path/to/model_1/",
        "model-2": "/path/to/model_2/",
    }

    # define configs
    # please see the config docstrings for more information about the config parameters
    configs = [
        InferenceConfig(
            data_path="/path/to/dataset.parquet",
            batch_size=128,
            heavy_column="sequence_aa_heavy",
            light_column="sequence_aa_light",
        ),
        PerPositionConfig(
            dataset_name="test",
            data_path="/path/to/dataset.csv",
            sequence_column="sequence_aa",
        ),
        MutationPredConfig(
            data_path=f"/path/to/airr-formatted-dataset.parquet",
            sequence_column="sequence_germ",
        ),
        ClassificationConfig(
            dataset_dir="/path/to/dataset",
            file_prefix="hd-0_cov-1",
            dataset_name="HD-CoV",
            sequence_column="sequence",
            num_folds=5,
            num_classes=2,
            epochs=3,
            train_batch_size=32,
            report_to="wandb",
            wandb_project="HD-CoV_classification",
        ),
        NaturalnessConfig(
            dataset_name="test",
            data_path=f"/path/to/dataset.csv",
            sequence_column="sequence_aa",
        ),
        # compatible with BALM MoE models only
        RoutingConfig(
            data_path=f"/path/to/dataset.csv",
            heavy_column="sequence_aa_heavy",
            light_column="sequence_aa_light",
        ),
    ]

    # run eval
    shared_output_dir = "./results"
    evaluate_ablms(
        models,
        configs,
        shared_output_dir,
        generate_comparisons=True,
    )

    ## if needed, you can regenerate comparisons as follows:
    # for all tasks
    compare_results(output_dir=shared_output_dir)
    # for a single task
    compare_task(
        task_type="mutation_prediction",
        task_results_dir="{shared_output_dir}/mutation_prediction/results/",
        output_dir="{shared_output_dir}/mutation_prediction/",
    )


if __name__ == "__main__":
    main()
