from ..compare_registry import register_comparer
from ...utils.tables import table_compare
from .inference_config import *
from .inference_run import *

register_comparer("inference", table_compare)
