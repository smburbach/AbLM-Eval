import pytest
from ablm_eval.tasks.compare_registry import register_comparer, _comparer_from_str


def dummy_compare(*args, **kwargs):
    return "ok"


def test_register_and_get_comparer():
    register_comparer("dummy", dummy_compare)
    assert _comparer_from_str("dummy") is dummy_compare


def test_comparer_from_str_invalid():
    with pytest.raises(ValueError):
        _comparer_from_str("not_a_task")
