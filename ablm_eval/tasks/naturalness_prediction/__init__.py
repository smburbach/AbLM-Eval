from ..compare_registry import register_comparer
from .naturalness_config import *
from .naturalness_plot import *
from .naturalness_run import *

register_comparer("naturalness", naturalness_compare)