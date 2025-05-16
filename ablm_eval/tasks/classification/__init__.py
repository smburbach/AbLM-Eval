from .classification_config import *
from .classification_run import *

from ..compare_registry import register_comparer
from ...utils.tables import table_compare

register_comparer("classification", table_compare)
