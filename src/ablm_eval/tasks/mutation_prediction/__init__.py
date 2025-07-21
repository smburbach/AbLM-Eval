from ..compare_registry import register_comparer
from .mutation_pred_config import *
from .mutation_pred_plot import *
from .mutation_pred_run import *

register_comparer("mutation_prediction", mut_pred_compare)
