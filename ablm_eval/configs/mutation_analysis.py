from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["MutationPredConfig"]


@dataclass
class MutationPredConfig:
    config_type: str = field(init=False, default="mutation_prediction")

    @property
    def task_dir(self):
        return "mutation_prediction"

    @property
    def name(self):
        return (self.task_dir).replace("_", " ").title()

    @property
    def runner(self):
        from ..tasks import run_mutation_analysis
        return run_mutation_analysis

    @property
    def comparer(self):
        pass

    # required
    mutation_data: str

    # output
    output_dir: str = None
