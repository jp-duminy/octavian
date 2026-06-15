from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import DataManager

import numpy as np
import pandas as pd

def wrap_positions(data_manager: DataManager) -> None:
  config = data_manager.config

  boxsize = data_manager.simulation['boxsize']/data_manager.simulation['h']

  # select halos with particles near boundary
  halos_to_wrap = {}
  for ptype in config['ptypes']:
    data_manager.load_property('pos', ptype)

  data = pd.concat([data_manager.data[ptype] for ptype in config['ptypes']])
  halos_grouped = data.groupby(by='HaloID', observed=True)
  for direction in ['x', 'y', 'z']:
    check_wrap = (halos_grouped[direction].max() - halos_grouped[direction].min()) > 0.5*boxsize
    halos_to_wrap[direction] = check_wrap[check_wrap].index.unique()

  # wrap positions
  for ptype in config['ptypes']:
    for direction in ['x', 'y', 'z']:
      in_halos_to_wrap = np.isin(data_manager.data[ptype]['HaloID'], halos_to_wrap[direction])
      too_high = data_manager.data[ptype][direction] > 0.5*boxsize

      data_manager.data[ptype].loc[in_halos_to_wrap & too_high, direction] -= boxsize