from ..compare_registry import register_comparer
from .routing_config import *
from .routing_run import *
from .routing_plot import *

register_comparer("routing_analysis", routing_compare)
