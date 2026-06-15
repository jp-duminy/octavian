import numpy as np
import pandas as pd
import h5py
import unyt
from astropy.cosmology import FlatLambdaCDM
from sympy import sympify


class DataManager:
  def __init__(self, snapfile: str, config: dict, comm=None):
    # mpi functionality
    self.comm = comm
    self.rank = comm.Get_rank() if comm else 0 # if statement allows for serial case
    self.size = comm.Get_size() if comm else 1

    self.snapfile = snapfile
    self.initialise_config(config)
    self.initialise_data()
    self.load_simulation_constants()

  # programmatic access to attrs based on https://peps.python.org/pep-0363/
  def __getitem__(self, name):
     return getattr(self, name)
  
  def __setitem__(self, name, value):
    return setattr(self, name, value)
  
  def __delitem__(self, name):
    return delattr(self, name)
  
  def __contains__(self, name):
     return hasattr(self, name)
  
  def initialise_config(self, config: dict) -> None:
    self.config = config

    with h5py.File(self.snapfile) as f:
      groups = list(f.keys())

    ptypes = []
    for ptype, ptype_name in self.config['ptype_names'].items():
      if ptype_name in groups:
        ptypes.append(ptype)

    self.config['ptypes'] = ptypes
    self.config['ptypes_baryon'] = [ptype for ptype in ptypes if ptype != 'dm']
    self.config['to_process'] = ptypes + ['baryon', 'total', 'local_densities', 'apertures']
    

  def initialise_data(self) -> None:
    self.data = {}
    
    ptypes = self.config['ptypes']
    
    for ptype in ptypes:
      self.data[ptype] = pd.DataFrame()

    

  def initialise_group_data(self) -> None:
    self.group_data = {}

    config = self.config

    ptypes = config['ptypes']
    groups = config['groups']
    groupIDs = config['groupIDs']

    for group in groups:
      ids = []
      for ptype in ptypes:
        id_column = groupIDs[group]
        ids.append(self.data[ptype][id_column].unique())

      ids = np.unique(np.concatenate(ids))
      if group == 'galaxies': ids = ids[ids != -1]

      self.group_data[group] = pd.DataFrame(index=ids)

  def load_simulation_constants(self) -> None:
    self.simulation = {}
    with h5py.File(self.snapfile) as f:
      header = f['Header'].attrs

      self.simulation['boxsize'] = header['BoxSize']
      self.simulation['O0'] = header['Omega0']
      self.simulation['Ol'] = header['OmegaLambda']
      self.simulation['Ok'] = 0 # header['Omegak']
      self.simulation['h'] = header['HubbleParam']
      self.simulation['redshift'] = header['Redshift']
      self.simulation['a'] = header['Time']

    self.simulation['G'] = unyt.G.to('cm**3/(g * s**2)')

    registry = unyt.UnitRegistry()
    registry.add('h', self.simulation['h'], sympify(1))
    registry.add('a', self.simulation['a'], sympify(1))

    self.units = unyt.UnitSystem('gadget', 'kpc', 'Msun', '1.e9 * yr', registry=registry)
    self.units['velocity'] = 'km/s'

    self.cosmology = FlatLambdaCDM(H0=100*self.simulation['h'], Om0=self.simulation['O0'])
    self.simulation['time_gyr'] = self.cosmology.age(self.simulation['redshift']).value
    self.simulation['time'] = (self.simulation['time_gyr'] * unyt.Gyr).to('s').d

    self.simulation['Hz'] = 100 * self.simulation['h'] * np.sqrt(self.simulation['Ol'] + self.simulation['O0'] * self.simulation['a']**-3) / (1 * unyt.kpc).to('m').d / (1 * unyt.s)
    self.simulation['rhocrit'] = (3. * self.simulation['Hz']**2 / (8. * np.pi * self.simulation['G'])).to('Msun / kpc**3').d

    self.simulation['E_z'] = np.sqrt(self.simulation['Ol'] + self.simulation['Ok'] * self.simulation['a']**-2 + self.simulation['O0'] * self.simulation['a']**-3)
    self.simulation['Om_z'] = self.simulation['O0'] * self.simulation['a']**-3 / self.simulation['E_z']**2

    self.simulation['r200_factor'] = (200 * 4./3. * np.pi*self.simulation['Om_z'] * self.simulation['rhocrit'] * self.simulation['a']**3)**(-1./3.)

  def create_unit_quantity(self, prop: str) -> unyt.unyt_quantity:
    return unyt.unyt_quantity(1., self.config['code_units'][prop], registry=self.units.registry)
  
  def get_prop_name(self, prop: str) -> str:
    return self.config['prop_aliases'][prop]

  def get_ptype_name(self, ptype: str) -> str:
    return self.config['ptype_names'][ptype]

  def get_column_name(self, prop: str) -> str | list[str]:
    try:
      column = self.config['prop_columns'][prop]
    except KeyError:
      column = prop

    return column
  
  def load_halo_ids(self):
    with h5py.File(self.snapfile) as f:
      for ptype in self.config['ptypes']:
        ptype_name = self.get_ptype_name(ptype)
        self.data[ptype]['HaloID'] = pd.Series(f[ptype_name]['HaloID'][:], dtype='category')
  
  def get_unit_conversion_factor(self, prop: str) -> float:
    try:
      data_units = self.config['prop_units'][prop]
      code_units = self.config['code_units'][prop]
      factor = unyt.unyt_quantity(1., data_units, registry=self.units.registry).to(code_units).d
      if prop == 'rho':
        factor *= self.config['XH'] / unyt.mp.to('g').d
    except KeyError:
      factor = 1.

    return factor

  def add_ptype_columns(self):
    for ptype in self.config['ptypes']:
      self.data[ptype]['ptype'] = pd.Series(np.full(len(self.data[ptype]), ptype), dtype='category')

  def load_property(self, prop: str, ptype: str):
    column = self.get_column_name(prop)
    prop_name = self.get_prop_name(prop)
    ptype_name = self.get_ptype_name(ptype)

    with h5py.File(self.snapfile) as f:
      if prop == 'metallicity':
        self.data[ptype][column] = f[ptype_name][prop_name][:, 0]
      elif prop == 'age':
        data = f[ptype_name][prop_name][:]
        self.data[ptype][column] = self.simulation['time_gyr'] - self.cosmology.age(1/data - 1).value
      else:
        self.data[ptype][column] = f[ptype_name][prop_name][:] * self.get_unit_conversion_factor(prop)
  