import pathlib

__all__ = ["create_dir"]


def create_dir(params):

    output_path = pathlib.Path(params.output_dir) / pathlib.Path(params.model_name)

    # check if path exists and is not empty
    if output_path.exists() and any(output_path.iterdir()):
        if params.model_name != "test_model":  # exception for testing
            raise Exception(
                f"The directory '{output_path}' already exists and is not empty!"
            )

    # make dir
    output_path.mkdir(parents=True, exist_ok=True)
