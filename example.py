from ablm_eval import (
    InferenceConfig,
    PerPositionConfig,
    MutationPredConfig,
    ClassificationConfig,
    RoutingConfig,
    NaturalnessConfig,
    DatasetColumns,
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
            dataset_name="paired-test",
            antibody_datatype="paired",
            data_path="/path/to/inference_data.parquet",
        ),
        PerPositionConfig(
            antibody_datatype="unpaired",
            data_path="/path/to/per_pos_data.parquet",
            dataset_columns=DatasetColumns(
                id_column="seq_id", chain_columns=["sequence"]
            ),
        ),
        MutationPredConfig(
            antibody_datatype="paired",
            data_path=f"/path/to/airr-formatted-dataset.parquet",
            data_processed=False,
            sequence_column="sequence_germ",
        ),
        ClassificationConfig(
            dataset_name="HD-Flu-CoV",
            data_path={
                i: {
                    "train": f"./class_data/hd-0_flu-1_cov-2_train{i}.csv",
                    "test": f"./class_data/hd-0_flu-1_cov-2_test{i}.csv",
                }
                for i in range(5)
            },
            antibody_datatype="paired",
            launcher="accelerate",
            dataset_columns=DatasetColumns(chain_columns=["h_sequence", "l_sequence"]),
            num_folds=5,
            num_classes=3,
            manually_freeze_base=True,
        ),
        NaturalnessConfig(
            antibody_datatype="unpaired",
            data_path=f"/path/to/dataset.csv",
        ),
        # compatible with BALM MoE models only
        RoutingConfig(
            antibody_datatype="paired",
            data_path=f"/path/to/dataset.csv",
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
