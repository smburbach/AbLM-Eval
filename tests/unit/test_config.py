import pytest
from ablm_eval import DatasetColumns, BaseDatasetConfig


# ### DatasetColumns tests
# def test_dataset_columns_apply_defaults_paired():
#     """Test apply_defaults method with paired datatype"""
#     columns = DatasetColumns()
#     columns.apply_defaults("paired")

#     assert columns.chain_names == ["heavy", "light"]
#     assert columns.chain_columns == ["sequence_aa_heavy", "sequence_aa_light"]
#     assert columns.cdr_columns == ["cdr_mask_heavy", "cdr_mask_light"]
#     assert columns.mutation_columns == [
#         "v_mutation_count_aa_heavy",
#         "v_mutation_count_aa_light",
#     ]


# def test_apply_defaults_unpaired(self):
#     """Test apply_defaults method with unpaired datatype"""
#     columns = DatasetColumns()
#     columns.apply_defaults("unpaired")

#     assert columns.chain_names == ["unpaired"]
#     assert columns.chain_columns == ["sequence_aa"]
#     assert columns.cdr_columns == ["cdr_mask"]
#     assert columns.mutation_columns == ["v_mutation_count_aa"]
#     assert columns.locus_column == "locus"


# def test_apply_defaults_preserves_existing_values(self):
#     """Test that apply_defaults only fills None values"""
#     columns = DatasetColumns(
#         chain_names=["custom_heavy"], chain_columns=None  # this should get filled
#     )
#     columns.apply_defaults("paired")

#     assert columns.chain_names == ["custom_heavy"]
#     assert columns.chain_columns == ["sequence_aa_heavy", "sequence_aa_light"]


# #### BaseDatasetConfig tests
# def test_missing_required_params(self):
#     """Test that missing required parameters raise TypeError"""
#     with pytest.raises(TypeError):
#         BaseDatasetConfig()
#     with pytest.raises(TypeError):
#         BaseDatasetConfig(data_path="data/test.parquet")
#     with pytest.raises(TypeError):
#         BaseDatasetConfig(antibody_datatype="paired")


# def test_invalid_antibody_datatype(self):
#     """Test that invalid antibody_datatype raises ValueError"""
#     with pytest.raises(ValueError, match="Invalid antibody_datatype"):
#         BaseDatasetConfig(data_path="data/test.parquet", antibody_datatype="invalid")


# def test_basic_configuration(self):
#     """Test basic valid configuration"""
#     config = BaseDatasetConfig(
#         data_path="data/test.parquet", antibody_datatype="paired"
#     )

#     assert config.data_path == "data/test.parquet"
#     assert config.antibody_datatype == "paired"
#     assert isinstance(config.dataset_columns, DatasetColumns)
#     assert config.dataset_columns.chain_names == ["heavy", "light"]
#     assert config.dataset_columns.chain_columns == [
#         "sequence_aa_heavy",
#         "sequence_aa_light",
#     ]


# def test_keep_columns_extension(self):
#     """Test that keep_columns is properly extended with required columns"""
#     config = BaseDatasetConfig(
#         data_path="data/test.parquet",
#         antibody_datatype="paired",
#         keep_columns=["extra_col"],
#     )

#     # Should include original column plus all the required ones
#     assert "extra_col" in config.keep_columns
#     assert "sequence_id" in config.keep_columns
#     assert "sequence_aa_heavy" in config.keep_columns
#     assert "sequence_aa_light" in config.keep_columns
