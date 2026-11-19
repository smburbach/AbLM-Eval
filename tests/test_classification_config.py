import yaml

from ablm_eval import ClassificationConfig


def test_accelerate_config_defaults_without_a_terminal(tmp_path):
    ClassificationConfig.override_accelerate_config("HD-CoV", tmp_path)
    written = yaml.safe_load((tmp_path / "HD-CoV_accelerate_config.yaml").read_text())
    assert written["distributed_type"] is not None
