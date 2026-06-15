from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from octavian.data_management import DataManager

import numpy as np
import pandas as pd
import unyt
from functools import partial
from astropy import constants as const

from scipy.spatial import KDTree

from octavian.aggregate_properties.group_computations import (
    compute_angular_momentum,
    compute_rotation_quantities,
    compute_radial_quantiles,
    compute_virial_quantities,
)

from octavian.aggregate_properties.group_helpers import (
    sum_per_group,
    count_per_group,
    max_value_per_group,
    max_idx_per_group,
    min_idx_per_group,
    broadcast_to_particles,
    sort_by_group,
    weighted_mean_per_group,
    extract_particle_arrays,
)

# Suppress pandas fragmented frame performance warnings (superfluous)  https://stackoverflow.com/a/76306267
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def common_group_properties(data_manager: DataManager, group_name: str, particle_type: str) -> None:
  """
  Computes properties common to all particle types.
  """
  config = data_manager.config
  groupID_key = config['groupIDs'][group_name]
  group_data = data_manager.group_data[group_name]
  n_groups = len(group_data)

  # -
  # step 1: extract arrays from datamanager so the entire operation can be vectorised
  # - 

  if particle_type == 'total':
      ptypes = config['ptypes']
  elif particle_type == 'baryon':
      ptypes = config['ptypes_baryon']
  else:
      ptypes = [particle_type]

  # ids: groups, halos, galaxies
  group_ids_list, halo_ids_list, gal_ids_list = [], [], []
  # physical quantities: mass, potential
  masses_list, potentials_list = [], []
  # kinematics: positions, velocities
  positions_list, velocities_list = [], []

  # cast the required data to numpy.
  for ptype in ptypes:
    df = data_manager.data[ptype]
    group_ids_list.append(df[groupID_key].to_numpy())
    halo_ids_list.append(df['HaloID'].to_numpy())
    gal_ids_list.append(df['GalID'].to_numpy())
    masses_list.append(df['mass'].to_numpy())
    potentials_list.append(df['potential'].to_numpy())
    positions_list.append(df[['x', 'y', 'z']].to_numpy())
    velocities_list.append(df[['vx', 'vy', 'vz']].to_numpy())

  # flat arrays
  group_ids = np.concatenate(group_ids_list)
  halo_ids = np.concatenate(halo_ids_list)
  gal_ids = np.concatenate(gal_ids_list)
  masses = np.concatenate(masses_list)
  potentials = np.concatenate(potentials_list)
  # vstack is equivalent to concatenate here: but these are 3D so this makes the format more explicit
  positions = np.vstack(positions_list)
  velocities = np.vstack(velocities_list)

  # filter particles not assigned to a specific galaxy
  if group_name == 'galaxies':
    keep = gal_ids != -1
    group_ids = group_ids[keep]
    halo_ids = halo_ids[keep]
    masses = masses[keep]
    potentials = potentials[keep]
    positions = positions[keep]
    velocities = velocities[keep]

  # if a group is empty
  if len(masses) == 0:
    return

  # -
  # step 2: group indexing
  # -

  group_idx = group_data.index.get_indexer(group_ids) # build a list of particles where the value of each particle is its group ID

  # HACK: builds from original HIDs then reassigns 
  # parent halo assignment
  if group_name == 'galaxies' and particle_type == 'total':
    parent = np.full(n_groups, -1, dtype=np.int64)
    for i in range(len(group_ids)):
        g = group_idx[i]
        if parent[g] == -1:
            parent[g] = halo_ids[i]
    group_data['parent_halo_index'] = parent

    max_hid = data_manager.group_data['halos'].index.max()
    halo_index_lookup = np.full(max_hid + 1, -1, dtype=np.int64)
    halo_index_lookup[data_manager.group_data['halos'].index] = np.arange(len(data_manager.group_data['halos']))
    parent = np.where(parent != -1, halo_index_lookup[parent], -1)
    group_data['parent_halo_index'] = parent

  # -
  # step 3: nparticles
  # - 

  counts = count_per_group(group_idx, n_groups) # number of particles per group
  group_data[f'n{particle_type}'] = counts

  # - 
  # step 4: masses
  # - 

  if particle_type == 'bh':
    group_mass = max_value_per_group(masses, group_idx, n_groups) # REVIEW: why?
    group_mass = np.where(np.isfinite(group_mass), group_mass, 0.0)
  else:
    group_mass = sum_per_group(masses, group_idx, n_groups)

  group_data[f'mass_{particle_type}'] = group_mass

  # -
  # step 5: minimum potential for halos
  # - 

  if group_name == 'halos' and particle_type == 'total':

    minimum_potential_idx = min_idx_per_group(potentials, group_idx, n_groups)
    valid = minimum_potential_idx >= 0 # mask all else away

    minimum_potential_position = np.full((n_groups, 3), np.nan)
    minimum_potential_velocity = np.full((n_groups, 3), np.nan)
    minimum_potential_position[valid] = positions[minimum_potential_idx[valid]]
    minimum_potential_velocity[valid] = velocities[minimum_potential_idx[valid]]

    for i, d in enumerate(['x', 'y', 'z']):
      group_data[f'minpot_{d}'] = minimum_potential_position[:, i]
      group_data[f'minpot_v{d}'] = minimum_potential_velocity[:, i]

  # - 
  # stage 6: centre-of-mass 
  # - 

  com_positions = np.zeros((n_groups, 3))
  com_velocities = np.zeros((n_groups, 3))

  for d in range(3):
    com_positions[:, d] = sum_per_group(positions[:, d] * masses, group_idx, n_groups) / group_mass
    com_velocities[:, d] = sum_per_group(velocities[:, d] * masses, group_idx, n_groups) / group_mass

  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'{d}_{particle_type}'] = com_positions[:, i]
  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'v{d}_{particle_type}'] = com_velocities[:, i]

  # -
  # stage 7: relative quantities
  # -

  # halos: potential well, galaxies: centre-of-mass
  if group_name == 'halos':
    if particle_type == 'total':
        ref_positions = minimum_potential_position
        ref_velocities = minimum_potential_velocity
    else:
        ref_positions = group_data[['minpot_x', 'minpot_y', 'minpot_z']].to_numpy()
        ref_velocities = group_data[['minpot_vx', 'minpot_vy', 'minpot_vz']].to_numpy()
  else:
      ref_positions = com_positions
      ref_velocities = com_velocities

  positions_rel = positions - broadcast_to_particles(ref_positions, group_idx)
  velocities_rel_com = velocities - broadcast_to_particles(com_velocities, group_idx)
  velocities_rel_ref = velocities - broadcast_to_particles(ref_velocities, group_idx)

  radii = np.linalg.norm(positions_rel, axis=1)

  # -
  # step 8: velocity dispersion
  # -

  disp_sums = sum_per_group(np.sum(velocities_rel_com**2, axis=1), group_idx, n_groups)
  vel_disps = np.where(counts > 0, np.sqrt(disp_sums / np.maximum(counts, 1)), np.nan)
  group_data[f'velocity_dispersion_{particle_type}'] = vel_disps

  # -
  # step 9: angular momentum and rotation
  # -

  L, ktot = compute_angular_momentum(positions_rel, velocities_rel_ref, masses, group_idx, n_groups)
  
  for i, d in enumerate(['x', 'y', 'z']):
    group_data[f'L{d}_{particle_type}'] = L[:, i]

  L_mag = np.linalg.norm(L, axis=1)
  group_data[f'L_{particle_type}'] = L_mag
  group_data[f'ALPHA_{particle_type}'] = np.arctan2(L[:, 1], L[:, 2])
  group_data[f'BETA_{particle_type}'] = np.arcsin(L[:, 0] / L_mag)

  counter_mass, krot, ktot = compute_rotation_quantities(
      positions_rel, velocities_rel_ref, masses, group_idx, L, n_groups
  )

  group_data[f'BoverT_{particle_type}'] = 2 * counter_mass / group_mass
  group_data[f'kappa_rot_{particle_type}'] = krot / ktot

  angular_cols = [
      f'velocity_dispersion_{particle_type}',
      f'Lx_{particle_type}', f'Ly_{particle_type}', f'Lz_{particle_type}',
      f'L_{particle_type}',
      f'ALPHA_{particle_type}', f'BETA_{particle_type}',
      f'BoverT_{particle_type}', f'kappa_rot_{particle_type}',
  ]
  # for small groups: set quantites = 0 as they are not meaningful (as done in original code)
  small = (counts > 0) & (counts < 3)
  for col in angular_cols:
      vals = group_data[col].to_numpy().copy()
      vals[small] = 0.
      group_data[col] = vals

  # -
  # step 10: radial quantities
  # -

  # from previous code
  quantiles = np.array([0.2, 0.5, 0.8])
  quantile_names = ['r20', 'half_mass', 'r80']

  radial_results = compute_radial_quantiles(radii, masses, group_idx, n_groups, quantiles)

  for q, col_name in enumerate(quantile_names):
      group_data[f'radius_{particle_type}_{col_name}'] = radial_results[:, q]

  # - 
  # step 11: virial quantities for halos
  # - 

  if group_name == 'halos' and particle_type == 'total':
    # from previous code
    group_data['r200'] = data_manager.simulation['r200_factor'] * group_mass**(1/3)
    G_factor = unyt.G.to('(km**2 * kpc)/(Msun * s**2)').d
    group_data['circular_velocity'] = np.sqrt(G_factor * group_mass / group_data['r200'])
    group_data['temperature'] = 3.6e5 * (group_data['circular_velocity'] / 100.0)**2
    group_data['spin_param'] = L_mag / (
        np.sqrt(2) * group_mass * group_data['circular_velocity'].to_numpy() * group_data['r200'].to_numpy()
    )

    rhocrit = (
        data_manager.simulation['rhocrit'] *
        data_manager.create_unit_quantity('rhocrit')
    ).to('Msun / (kpc*a)**3').d

    factors = np.array([200., 500., 2500.])

    virial_r, virial_m = compute_virial_quantities(radii, masses, group_idx, n_groups, rhocrit, factors)

    for f, factor in enumerate([200, 500, 2500]):
        group_data[f'radius_{factor}_c'] = virial_r[:, f]
        group_data[f'mass_{factor}_c'] = virial_m[:, f]

  # NOTE: blanket pass at the end to enforce NaN vs zero convention.
  empty = counts == 0
  if np.any(empty):
      nan_cols = [
          f'velocity_dispersion_{particle_type}',
          f'Lx_{particle_type}', f'Ly_{particle_type}', f'Lz_{particle_type}',
          f'L_{particle_type}',
          f'ALPHA_{particle_type}', f'BETA_{particle_type}',
          f'BoverT_{particle_type}', f'kappa_rot_{particle_type}',
          f'radius_{particle_type}_r20', f'radius_{particle_type}_half_mass', f'radius_{particle_type}_r80',
      ]
      for col in nan_cols:
          if col in group_data.columns:
              vals = group_data[col].to_numpy().copy()
              vals[empty] = np.nan
              group_data[col] = vals

def gas_group_properties(data_manager: DataManager, group_name: str) -> None:
    
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  # load relevant quantities
  df = data_manager.data['gas']
  group_ids = df[groupID_key].to_numpy()
  masses = df['mass'].to_numpy()
  metallicities = df['metallicity'].to_numpy()
  sfrs = df['sfr'].to_numpy()
  temperatures = df['temperature'].to_numpy()
  rhos = df['rho'].to_numpy()
  mass_HI = df['mass_HI'].to_numpy()
  mass_H2 = df['mass_H2'].to_numpy()

  if group_name == 'galaxies':
    keep = df['GalID'].to_numpy() != -1
    group_ids = group_ids[keep]
    masses = masses[keep]
    metallicities = metallicities[keep]
    sfrs = sfrs[keep]
    temperatures = temperatures[keep]
    rhos = rhos[keep]
    mass_HI = mass_HI[keep]
    mass_H2 = mass_H2[keep]

  # guard against empty group
  if len(masses) == 0:
    return
  
  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)

  # HI, H2 masses
  group_data['mass_HI'] = sum_per_group(mass_HI, group_idx, n_groups)
  group_data['mass_H2'] = sum_per_group(mass_H2, group_idx, n_groups)

  # SFR
  group_data['sfr'] = sum_per_group(sfrs, group_idx, n_groups)

  # metallicity
  group_mass = group_data[f'mass_gas'].to_numpy()
  group_sfr = group_data['sfr'].to_numpy()

  group_data['metallicity_mass_weighted'] = sum_per_group(metallicities * masses, group_idx, n_groups) / group_mass
  group_data['metallicity_sfr_weighted'] = sum_per_group(metallicities * sfrs, group_idx, n_groups) / group_sfr

  # CGM quantities
  cgm_mask = rhos < config['nHlim']
  cgm_idx = group_idx[cgm_mask]
  cgm_masses = masses[cgm_mask]
  cgm_temps = temperatures[cgm_mask]
  cgm_metals = metallicities[cgm_mask]

  group_data['mass_cgm'] = sum_per_group(cgm_masses, cgm_idx, n_groups)
  cgm_mass = group_data['mass_cgm'].to_numpy()

  # temperatures
  temp_mass_weighted = sum_per_group(temperatures * masses, group_idx, n_groups)
  group_data['temp_mass_weighted'] = temp_mass_weighted / group_mass

  cgm_temp_mass = sum_per_group(cgm_temps * cgm_masses, cgm_idx, n_groups)
  cgm_temp_metal = sum_per_group(cgm_temps * cgm_masses * cgm_metals, cgm_idx, n_groups)

  group_data['temp_mass_weighted_cgm'] = cgm_temp_mass / cgm_mass
  group_data['temp_metal_weighted_cgm'] = cgm_temp_metal / cgm_temp_mass

  # CGM metallicity
  cgm_metal_mass = sum_per_group(cgm_metals * cgm_masses, cgm_idx, n_groups)
  group_data['metallicity_mass_weighted_cgm'] = cgm_metal_mass / cgm_mass
  group_data['metallicity_temp_weighted_cgm'] = cgm_temp_metal / cgm_metal_mass

def star_group_properties(data_manager: DataManager, group_name: str) -> None:
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  df = data_manager.data['star']
  group_ids = df[groupID_key].to_numpy()
  masses = df['mass'].to_numpy()
  metallicities = df['metallicity'].to_numpy()
  ages = df['age'].to_numpy()

  if group_name == 'galaxies':
      keep = df['GalID'].to_numpy() != -1
      group_ids = group_ids[keep]
      masses = masses[keep]
      metallicities = metallicities[keep]
      ages = ages[keep]

  if len(masses) == 0:
      return

  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)

  # metallicity
  metal_mass = sum_per_group(metallicities * masses, group_idx, n_groups)
  total_mass = sum_per_group(masses, group_idx, n_groups)
  group_data['metallicity_stellar'] = metal_mass / total_mass

  # age
  group_mass_star = group_data['mass_star'].to_numpy()
  group_data['age_mass_weighted'] = sum_per_group(ages * masses, group_idx, n_groups) / group_mass_star
  group_data['age_metal_weighted'] = sum_per_group(ages * masses * metallicities, group_idx, n_groups) / metal_mass

def bh_group_properties(data_manager: DataManager, group_name: str) -> None:
  config = data_manager.config
  group_data = data_manager.group_data[group_name]
  groupID_key = config['groupIDs'][group_name]

  df = data_manager.data['bh']
  group_ids = df[groupID_key].to_numpy()
  masses = df['mass'].to_numpy()
  bhmdots = df['bhmdot'].to_numpy()

  if group_name == 'galaxies':
      keep = df['GalID'].to_numpy() != -1
      group_ids = group_ids[keep]
      masses = masses[keep]
      bhmdots = bhmdots[keep]

  if len(masses) == 0:
      return

  n_groups = len(group_data)
  group_idx = group_data.index.get_indexer(group_ids)
  
  # find most massive BH per group
  max_idx = max_idx_per_group(masses, group_idx, n_groups)
  valid = max_idx >= 0

  # bhmdot of most massive BH
  bhmdot = np.full(n_groups, np.nan)
  bhmdot[valid] = bhmdots[max_idx[valid]]
  group_data['bhmdot'] = bhmdot

  # Eddington fraction
  FRAD = 0.1
  edd_factor = (4 * np.pi * const.G * const.m_p / (FRAD * const.c * const.sigma_T)).to('1/yr').value

  bh_mass = np.full(n_groups, np.nan)
  bh_mass[valid] = masses[max_idx[valid]]
  group_data['bh_fedd'] = bhmdot / (edd_factor * bh_mass)

def calculate_local_densities(data_manager: DataManager) -> None:

  config = data_manager.config
  groups = config['groups']

  for group in groups:
    group_data = data_manager.group_data[group]

    if len(group_data) == 0:
      continue

    pos = group_data[['x_total', 'y_total', 'z_total']].to_numpy()
    mass = group_data['mass_total'].to_numpy()

    # REVIEW: moving a FOF6D optimisation into this function
    # previously the pandas .explode() calls were memory-intensive
    tree = KDTree(pos)
    for radius in [300., 1000., 3000.]:
      volume = 4./3. * np.pi * radius**3
      index_lists = tree.query_ball_point(pos, radius, workers=-1) # workers=-1 means all processors are used (from documentation)
      mass_sums = np.array([mass[il].sum() for il in index_lists])
      counts = np.array([len(il) for il in index_lists])

      group_data[f'local_mass_density_{int(radius)}'] = mass_sums / volume
      group_data[f'local_number_density_{int(radius)}'] = counts / volume

def calculate_aperture_masses(data_manager, config):
    
    group_data = data_manager.group_data['galaxies']
    n_galaxies = len(group_data)
    galaxy_pos = group_data[['x_total', 'y_total', 'z_total']].to_numpy()
    parent_halo = group_data['parent_halo_index'].to_numpy()
    aperture = 30. # as defined previously

    # use helper function
    all_pos, all_mass, all_codes, all_halos, ptype_names = extract_particle_arrays(
        data_manager, config, include_hydrogen=True
    )
    # may want to check the helper function include_hydrogen part, thought it might be useful in future
    n_ptypes = len(ptype_names)

    # pre-sort particles by halo
    order, unique_halos, h_start, h_end = sort_by_group(all_halos)
    all_pos = all_pos[order]
    all_mass = all_mass[order]
    all_codes = all_codes[order]

    # pre-sort galaxies by parent halo
    gal_order, halos_with_galaxies, gal_start, gal_end = sort_by_group(parent_halo)

    result = np.zeros((n_galaxies, n_ptypes))

    for h in range(len(unique_halos)):
        halo_id = unique_halos[h]
        halo_pos = all_pos[h_start[h]:h_end[h]]
        halo_mass = all_mass[h_start[h]:h_end[h]]
        halo_codes = all_codes[h_start[h]:h_end[h]]

        # guard
        if len(halo_pos) == 0:
            continue

        gh_idx = np.searchsorted(halos_with_galaxies, halo_id) # find where the hid sits in halos_with_galaxies
        if gh_idx >= len(halos_with_galaxies) or halos_with_galaxies[gh_idx] != halo_id:
            continue # skip halos with no galaxies
        gal_indices = gal_order[gal_start[gh_idx]:gal_end[gh_idx]]

        # guard
        if len(gal_indices) == 0:
            continue

        # build KDTree (explained in FOF6D code)
        tree = KDTree(halo_pos)
        neighbor_lists = tree.query_ball_point(galaxy_pos[gal_indices], aperture)

        for galaxies_idx_local, neighbours in enumerate(neighbor_lists):
            if len(neighbours) == 0:
                continue
            neighbours = np.array(neighbours)
            # np.bincount optimisation, as in the group_helpers.py functions
            masses_by_type = np.bincount(
                halo_codes[neighbours], weights=halo_mass[neighbours], minlength=n_ptypes
            )
            # masses contained within the 30kpc aperture
            result[gal_indices[galaxies_idx_local], :] = masses_by_type

    for i, name in enumerate(ptype_names):
        group_data[f'mass_{name}_30kpc'] = result[:, i]

    group_data['mass_total_30kpc'] = result[:, :len(config['ptypes'])].sum(axis=1)

def calculate_group_properties(data_manager: DataManager) -> None:

  # admin
  config = data_manager.config
  for ptype in config['ptypes']:
    data_manager.load_property('potential', ptype)

  groups = config['groups']

  # unnecessary columns
  columns_to_drop = ['vx', 'vy', 'vz', 'potential']
  to_process = config['to_process']

  # order to iterate over
  ptype_order = ['total', 'dm', 'baryon', 'gas', 'star', 'bh']
  # drop the necessary columns afterwards
  drop_after = {'dm', 'gas', 'star', 'bh'}

  # common group properties
  for ptype in ptype_order:
      if ptype not in to_process:
          continue
      for group in groups:
          common_group_properties(data_manager, group, ptype)
      if ptype in drop_after:
          data_manager.data[ptype].drop(columns=columns_to_drop, inplace=True)

  # gas properties
  if 'gas' in to_process:
    for property in ['rho', 'nh', 'fH2', 'metallicity', 'sfr', 'temperature']:
      data_manager.load_property(property, 'gas')

    # gas masses
    data = data_manager.data['gas']
    data['fHI'] = data.eval('nh / mass')
    not_conserving_mass = data.eval('(fHI + fH2) > 1')
    data.loc[not_conserving_mass, 'fHI'] = 1. - data.loc[not_conserving_mass, 'fH2']

    data['mass_HI'] = config['XH'] * data['fHI'] * data['mass']
    data['mass_H2'] = config['XH'] * data['fH2'] * data['mass']

    for group in groups:
      gas_group_properties(data_manager, group)

  # star properties
  if 'star' in to_process:
    for property in ['age', 'metallicity']:
      data_manager.load_property(property, 'star')

    for group in groups:
      star_group_properties(data_manager, group)

  # bh properties
  if 'bh' in to_process:
    for property in ['bhmdot']:
      data_manager.load_property(property, 'bh')
      
    for group in groups:
      bh_group_properties(data_manager, group)

  # apertures
  if 'apertures' in to_process and 'galaxies' in config['groups']:
    calculate_aperture_masses(data_manager, config)

  # densities
  if 'local_densities' in to_process:
    calculate_local_densities(data_manager)
