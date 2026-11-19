import re

import pytest

from ablm_eval.lab_models import LAB_MODELS, resolve_model

TINY = {
    "name": "tiny",
    "path": "brineylab/8M_full_checkpoint-500000",
    "revision": "01e6532f909186a7f32ffd3e5fd6ced80daa0cc1",
    "separator": "<cls><cls>",
}


def test_registry_holds_the_nine_lab_models():
    assert set(LAB_MODELS) == {
        "balm-paired",
        "balm-unpaired",
        "ft-esm",
        "balm-moe",
        "currab",
        "preferential-250k",
        "uniform-250k",
        "uniform-350k",
        "scaling-350m-full",
    }


def test_every_revision_is_a_full_sha():
    assert all(re.fullmatch(r"[0-9a-f]{40}", m.revision) for m in LAB_MODELS.values())


def test_unpaired_models_have_no_separator():
    assert LAB_MODELS["balm-unpaired"].separator is None


@pytest.mark.parametrize(
    ("name", "separator"),
    [
        ("balm-paired", "</s>"),
        ("ft-esm", "<cls><cls>"),
        ("balm-moe", "<cls>"),
        ("currab", "<cls>"),
        ("preferential-250k", "<cls><cls>"),
        ("uniform-250k", "<cls><cls>"),
        ("uniform-350k", "<cls><cls>"),
        ("scaling-350m-full", "<cls><cls>"),
    ],
)
def test_separators_match_training(name, separator):
    assert resolve_model(name).separator == separator


def test_resolve_full_entry():
    m = resolve_model(TINY)
    assert m.repo == "brineylab/8M_full_checkpoint-500000"
    assert m.name == "tiny"


def test_full_entry_null_separator_means_unpaired():
    assert resolve_model({**TINY, "separator": None}).separator is None


def test_full_entry_without_separator_is_refused():
    entry = {k: v for k, v in TINY.items() if k != "separator"}
    with pytest.raises(ValueError, match="separator"):
        resolve_model(entry)


@pytest.mark.parametrize("rev", ["main", "01e6532f"])
def test_resolve_refuses_unpinned_revision(rev):
    with pytest.raises(ValueError, match="revision"):
        resolve_model({**TINY, "revision": rev})


def test_resolve_unknown_name_names_it():
    with pytest.raises(ValueError, match="no-such-model"):
        resolve_model("no-such-model")


def test_model_name_with_a_slash_is_refused():
    with pytest.raises(ValueError, match="org/model"):
        resolve_model({**TINY, "name": "org/model"})
