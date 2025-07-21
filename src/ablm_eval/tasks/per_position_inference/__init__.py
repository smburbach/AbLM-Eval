from ..compare_registry import register_comparer
from .per_position_inference_config import *
from .per_position_inference_run import *
from .per_position_plot import *

register_comparer("per_pos_inference", per_pos_compare)
