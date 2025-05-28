import pytest

from .utils import generate_mini_models


@pytest.fixture(scope="session")
def mini_models(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("mini_models")
    return generate_mini_models(str(output_dir))
