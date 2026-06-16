from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import DataManager

import numpy as np
import pandas as pd

# FIXME: worst function in the database, highly inefficient and destructively mutates coordinates incorrectly
  # remove and move to unwrapping relative to centres.

def wrap_positions(data_manager: DataManager) -> None:
  config = data_manager.config
  
  for ptype in config['ptypes']:
      data_manager.load_property('mass', ptype)
      data_manager.load_property('pos', ptype)
      data_manager.load_property('vel', ptype)