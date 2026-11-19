"""The lab's published antibody language models, pinned to a revision with their separators."""

import re
from dataclasses import dataclass

__all__ = ["LAB_MODELS", "LabModel", "resolve_model"]

_FULL_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class LabModel:
    """A model on the Hugging Face Hub, pinned to one commit.

    Attributes:
        name: Short name used in configs and results.
        repo: Hugging Face repository id.
        revision: Full 40-character commit SHA of the repository.
        separator: Text joining the heavy and light chains, as in training.
            ``None`` means the model was trained on single chains only.
    """

    name: str
    repo: str
    revision: str
    separator: str | None

    def __post_init__(self) -> None:
        if not _FULL_SHA.fullmatch(self.revision):
            raise ValueError(
                f"model {self.name!r}: revision {self.revision!r} is not a full commit SHA"
            )


LAB_MODELS: dict[str, LabModel] = {
    m.name: m
    for m in [
        LabModel(
            "balm-paired",
            "brineylab/BALM-paired",
            "78e6a59276dbcf7ccd4440091eb29acf8b81062d",
            "</s>",
        ),
        LabModel(
            "balm-unpaired",
            "brineylab/BALM-unpaired",
            "8fd18a5b47bf6c736a031f8f1d76d779653ebe8a",
            None,
        ),
        LabModel(
            "ft-esm", "brineylab/ft-ESM", "6594c8b61de11d6cfe2e6e7a00b380b19ba6a0a7", "<cls><cls>"
        ),
        LabModel(
            "balm-moe", "brineylab/BALM-MoE", "51842e69ca3f0ae3f239d5fbd854c78bc0857f82", "<cls>"
        ),
        LabModel("currab", "brineylab/CurrAb", "92e28534663e163f1b398f773b3fe041085737d9", "<cls>"),
        LabModel(
            "preferential-250k",
            "brineylab/preferential-250k",
            "72ebb7b3e0f126407f1c6175e66f0cde1cb11fdc",
            "<cls><cls>",
        ),
        LabModel(
            "uniform-250k",
            "brineylab/uniform-250k",
            "bb2fd2338300d70da77f5df8db46ac8ac0f31a5f",
            "<cls><cls>",
        ),
        LabModel(
            "uniform-350k",
            "brineylab/uniform-350k",
            "43a1e4fd35cf88ca208caaab2089766e6bc93be1",
            "<cls><cls>",
        ),
        LabModel(
            "scaling-350m-full",
            "brineylab/350M_full_checkpoint-500000",
            "9cd9e8af5138ff5da4166585985a0fe5c6afe330",
            "<cls><cls>",
        ),
    ]
}


def resolve_model(entry: str | dict) -> LabModel:
    """Turn a config's model entry into a pinned model.

    Args:
        entry: A lab model's name, or a mapping with ``name``, ``path`` (the
            Hugging Face repository), ``revision`` and ``separator``.

    Returns:
        The pinned model.

    Raises:
        ValueError: If the name is not a lab model, a key is missing, or the
            revision is not a full commit SHA.
    """
    if isinstance(entry, str):
        if entry not in LAB_MODELS:
            raise ValueError(f"unknown model {entry!r}; lab models: {', '.join(LAB_MODELS)}")
        return LAB_MODELS[entry]
    missing = [k for k in ("name", "path", "revision", "separator") if k not in entry]
    if missing:
        raise ValueError(
            f"model entry {entry.get('name', entry)!r} is missing {', '.join(missing)}"
        )
    return LabModel(entry["name"], entry["path"], entry["revision"], entry["separator"])
