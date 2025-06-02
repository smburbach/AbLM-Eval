import pytest
from ablm_eval import ClassificationConfig, PerPositionConfig


def test_classification_config_init():
    config = ClassificationConfig(
        dataset_dir="data/",
        file_prefix="prefix",
        dataset_name="test",
        num_classes=2,
        learning_rate=0.01,
    )
    assert config.dataset_dir == "data/"
    assert config.num_classes == 2


# def test_per_position_config_validation():
#     # Should raise if missing heavy_column or light_column
#     with pytest.raises(ValueError):
#         PerPositionConfig(
#             data_path="data/test_data.parquet",
#             dataset_name="test",
#             heavy_column=None,
#             light_column="L",
#         )
