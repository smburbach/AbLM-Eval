import os


# temporary test github actions
def test_basic():
    assert 1 + 1 == 2


def test_mini_models_created(mini_models):
    # check mini model creation
    expected_models = ["BALM-dense", "BALM-MoE", "ESM"]
    for model_name in expected_models:
        # check for file path
        path = mini_models.get(model_name)
        assert (
            path is not None
        ), f"Model path for {model_name} not found in mini_models."

        # check that the directory exists and contains a config.json
        assert os.path.isdir(
            path
        ), f"Model directory for {model_name} does not exist: {path}"
        assert os.path.isfile(
            os.path.join(path, "config.json")
        ), f"config.json missing for {model_name} at {path}"


# check compare_results fn
# check compare-task fn
