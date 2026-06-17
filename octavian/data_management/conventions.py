"""

Octavian units/dtype conventions.

"""

# default packages
from dataclasses import dataclass
from astropy.constants import codata2014 as codata # unyt uses codata2014, need to migrate to codata2022
from astropy.constants import iau2015 as iau 
import astropy.units as u
from astropy.cosmology import FLRW
import numpy as np

@dataclass(frozen=True, slots=True)
class PhysicalConstants:
    """
    From CODATA2014 and IAU2015 (plan to update to CODATA2022).
    """
    G_CGS:          float = codata.G.cgs.value
    C_CGS:          float = codata.c.cgs.value
    PROTON_MASS_G:  float = codata.m_p.cgs.value
    BOLTZMANN_CGS:  float = codata.k_B.cgs.value
    SIGMA_T_CGS:    float = codata.sigma_T.cgs.value
    M_SUN_G:        float = iau.M_sun.cgs.value
    KPC_CM:         float = iau.kpc.cgs.value
    KPC_M:          float = iau.kpc.si.value
    GYR_S:          float = (1* u.Gyr).to(u.s).value

    HUBBLE_UNIT:          float = (1 * u.km / u.s / u.Mpc).to(1/u.s).value
    RHO_CGS_TO_MSUN_KPC3: float = (1 * u.g / u.cm**3).to(u.M_sun / u.kpc**3).value
    G_VCIRC:              float = codata.G.to('km**2 * kpc / (M_sun * s**2)').value

CONSTANTS = PhysicalConstants()

CODE_UNITS = {

    "pos":              "kpc * a", # comoving
    "vel":              "km/s", # peculiar
    "mass":             "Msun",
    "temperature":      "K",
    "rho":              "g/cm**3",
    "rhocrit":          "Msun/kpc**3",
    "sfr":              "Msun/yr",
    "bhmass":           "Msun",
    "bhmdot":           "Msun/yr",

}

DTYPES = {

    "pid":              np.int64,
    "ptype":            np.int8, # this allows up to 256 ptypes 

    # NOTE: quantities requiring 64-bit precision
    "pos":              np.float64,
    "vel":              np.float64,
    "mass":             np.float64,
    "rho":              np.float64,
    "internal_energy":  np.float64,
    "sfr":              np.float64,
    "metallicity":      np.float64,
    "nh":               np.float64,
    "fH2":              np.float64,
    "potential":        np.float64,
    "formation_time":   np.float64,
    "bhmass":           np.float64,
    "bhmdot":           np.float64,

    # NOTE: external halo readers (AHF) sometimes store absurdly large integers for HIDs
    "HaloID":           np.int64,
    "GalID":            np.int64,
    "parent_halo":      np.int64,

    "ngas":             np.int32, # largest halo in simba had 19mil particles; this should suffice
    "nstar":            np.int32,
    "ndm":              np.int32,
    "nbh":              np.int32,

    "csr_offsets":      np.int64,
    "csr_lengths":      np.int32, # same justification as for nparticles
    "csr_indices":      np.int64,

}

def gizmo_unit_conversion_factor(dataset: str, h: float, a: float) -> float:
    """
    Gizmo snapshot unit conversion factors.

    Please see http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html#snaps-units

    Astropy for automatic dimensional analysis.
    """
    conversions = {
        "pos":             (u.kpc * a / h,                                  u.kpc * a),
        "vel":             (u.km / u.s * np.sqrt(a),                        u.km / u.s),
        "mass":            (1e10 * u.M_sun / h,                             u.M_sun),
        "rho":             (1e10 * u.M_sun * h**2 / (u.kpc**3 * a**3),      u.g / u.cm**3),
        "internal_energy": (u.km**2 / u.s**2,                               u.km**2 / u.s**2),
        "sfr":             (u.M_sun / u.yr,                                 u.M_sun / u.yr),
        "metallicity":     (u.dimensionless_unscaled,                       u.dimensionless_unscaled),
        "nh":              (u.dimensionless_unscaled,                       u.dimensionless_unscaled),
        "fH2":             (u.dimensionless_unscaled,                       u.dimensionless_unscaled),
        "potential":       (u.km**2 / u.s**2,                               u.km**2 / u.s**2),
        "formation_time":  (u.dimensionless_unscaled,                       u.dimensionless_unscaled),
        "bhmass":          (1e10 * u.M_sun / h,                             u.M_sun),
        "bhmdot":          (10.2249488753 * u.M_sun / (h * u.yr),           u.M_sun / u.yr),
    }

    if dataset not in conversions:
        raise KeyError(f"{dataset} is not recognised in the unit conversions.")

    snap_unit, internal_unit = conversions[dataset]
    return (1 * snap_unit).to(internal_unit).value

def derive_stellar_age(formation_time: np.ndarray, time_gyr: float, cosmology: FLRW) -> np.ndarray:
    """
    Converts GIZMO stellar formation time into stellar age in GYr.
    """
    redshifts = 1.0 / formation_time - 1.0
    return time_gyr - cosmology.age(redshifts).value # see astropy for integration details

def calculate_hydrogen_number_density(rho_cgs: np.ndarray, XH: float) -> np.ndarray:
    """
    Calculates nh from the simulation parameters. 
    """
    return rho_cgs * XH / CONSTANTS.PROTON_MASS_G

# TODO: temperature from internal energy.