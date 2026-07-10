"""

Octavian units/dtype conventions and conversions.
Also contains backend dataclasses.

"""

# default packages
from dataclasses import dataclass, field
from yaml import safe_load
from pathlib import Path

# units/arrays
from astropy.constants import codata2014 as codata  # unyt uses codata2014, need to migrate to codata2022
from astropy.constants import iau2015 as iau
import astropy.units as u
from astropy.cosmology import FLRW
import numpy as np

from octavian.log import get_logger

logger = get_logger()


@dataclass(frozen=True, slots=True)
class OctavianConfig:
    """
    User-configured parameters as specified in config.yaml.
    """

    stages: dict[str, bool]
    process_ptypes: dict[str, bool]

    min_stars_per_galaxy: int
    min_dm_per_halo: int

    nH_lim: float
    T_lim: float
    XH: float
    FRAD: float
    MU: float

    b: float
    velocity_factor: float

    radial_quantiles: dict[str, float]
    aperture_size: list[int]
    virial_factors: list[int]
    density_radii: list[int]

    cores_per_rank: int
    fof6d_weight: float
    properties_weight: float

    terminal_output_level: str

    @classmethod
    def from_yaml(cls, config_path: Path) -> OctavianConfig:
        """
        Parses a config.yaml file into the dataclass.
        """
        with open(config_path, "r") as f:
            raw = safe_load(f)

        return cls(
            stages=raw["stages"],
            process_ptypes=raw["process_ptypes"],
            min_stars_per_galaxy=raw["MINIMUM_STARS_PER_GALAXY"],
            min_dm_per_halo=raw["MINIMUM_DM_PER_HALO"],
            nH_lim=raw["nH_lim"],
            T_lim=raw["T_lim"],
            XH=raw["XH"],
            FRAD=raw["FRAD"],
            MU=raw["MU"],
            b=raw["b"],
            velocity_factor=raw["velocity_factor"],
            radial_quantiles=raw["radial_quantiles"],
            aperture_size=raw["aperture_size"],
            virial_factors=raw["virial_factors"],
            density_radii=raw["density_radii"],
            cores_per_rank=raw["cores_per_rank"],
            fof6d_weight=raw["fof6d_weight"],
            properties_weight=raw["properties_weight"],
            terminal_output_level=raw["terminal_output_level"],
        )


@dataclass(frozen=True, slots=True)
class OctavianConstants:
    """
    From CODATA2014 and IAU2015 (plan to update to CODATA2022).
    """

    # config-dependent parameters
    mu: float = 0.6  # (assuming X=0.7, Y=0.28, Z=0.02)
    frad: float = 0.1

    # fundamental values (CODATA/IAU)
    G_CGS: float = codata.G.cgs.value
    C_CGS: float = codata.c.cgs.value
    PROTON_MASS_G: float = codata.m_p.cgs.value
    BOLTZMANN_CGS: float = codata.k_B.cgs.value
    SIGMA_T_CGS: float = codata.sigma_T.cgs.value
    M_SUN_G: float = iau.M_sun.cgs.value
    KPC_CM: float = iau.kpc.cgs.value
    KPC_M: float = iau.kpc.si.value
    GYR_S: float = (1 * u.Gyr).to(u.s).value

    # derived unit conversions
    HUBBLE_UNIT: float = (1 * u.km / u.s / u.Mpc).to(1 / u.s).value
    RHO_CGS_TO_MSUN_KPC3: float = (1 * u.g / u.cm**3).to(u.M_sun / u.kpc**3).value
    G_VCIRC: float = codata.G.to(u.km**2 * u.kpc / (u.M_sun * u.s**2)).value

    # derived factors
    VIRIAL_TEMP_FACTOR: float = field(init=False)
    EDD_FACTOR: float = field(init=False)

    def __post_init__(self) -> None:

        object.__setattr__(  # expect ~3.6e5
            self,
            "VIRIAL_TEMP_FACTOR",
            self.mu * codata.m_p.cgs.value * u.km.to(u.cm) ** 2 / (2 * codata.k_B.cgs.value),
        )

        object.__setattr__(  # expect ~2.2e-8
            self,
            "EDD_FACTOR",
            (4 * np.pi * codata.G.cgs * codata.m_p.cgs / (self.frad * codata.c.cgs * codata.sigma_T.cgs))
            .to(1 / u.yr)
            .value,
        )


@dataclass(slots=True, frozen=True)
class SimulationAttributes:
    """
    Simulation attributes (e.g. hubble parameter), read/derived from the header information.
    """

    # header attributes
    boxsize: float
    h: float
    a: float
    redshift: float
    omega_matter: float  # I renamed this because O0 isn't very readable
    omega_lambda: float
    mis: float

    # derived
    cosmology: FLRW
    time_gyr: float
    time: float
    Hz: float
    rhocrit: float
    rhocrit_comoving: float
    E_z: float
    omega_matter_z: float
    r200_factor: float


CODE_UNITS = {
    "pos": "kpc * a",  # comoving
    "vel": "km/s",  # peculiar
    "mass": "Msun",
    "temperature": "K",
    "rho": "g/cm**3",
    "rhocrit": "Msun/kpc**3",
    "sfr": "Msun/yr",
    "bhmass": "Msun",
    "bhmdot": "Msun/yr",
}

DTYPES = {
    "pid": np.int64,
    "ptype": np.int8,  # this allows up to 256 ptypes
    # NOTE: quantities requiring 64-bit precision
    "pos": np.float64,
    "vel": np.float64,
    "mass": np.float64,
    "rho": np.float64,
    "internal_energy": np.float64,
    "sfr": np.float64,
    "metallicity": np.float64,
    "fHI": np.float64,
    "fH2": np.float64,
    "potential": np.float64,
    "formation_time": np.float64,
    "bhmass": np.float64,
    "bhmdot": np.float64,
    # NOTE: external halo readers (AHF) sometimes store absurdly large integers for HIDs
    "HaloID": np.int64,
    "GalID": np.int64,
    "parent_halo": np.int64,
    "ngas": np.int32,  # largest halo in simba had 19mil particles; this should suffice
    "nstar": np.int32,
    "ndm": np.int32,
    "nbh": np.int32,
    "csr_offsets": np.int64,
    "csr_lengths": np.int32,  # same justification as for nparticles
    "csr_indices": np.int64,
}


def gizmo_unit_conversion_factor(dataset: str, h: float, a: float) -> float:
    """
    Gizmo snapshot unit conversion factors (pulled from config.yaml)
    Please see http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html#snaps-units
    Astropy for automatic dimensional analysis.
    """
    conversions = {
        "pos": (u.kpc * a / h, u.kpc * a),
        "vel": (u.km / u.s * np.sqrt(a), u.km / u.s),
        "mass": (1e10 * u.M_sun / h, u.M_sun),  # 1e10 factor is just a Gizmo scaling convention
        "rho": (1e10 * u.M_sun * h**2 / (u.kpc**3 * a**3), u.g / u.cm**3),
        "internal_energy": (u.km**2 / u.s**2, u.cm**2 / u.s**2),
        "sfr": (u.M_sun / u.yr, u.M_sun / u.yr),
        "metallicity": (u.dimensionless_unscaled, u.dimensionless_unscaled),
        "fHI": (u.dimensionless_unscaled, u.dimensionless_unscaled),
        "fH2": (u.dimensionless_unscaled, u.dimensionless_unscaled),
        "potential": (u.km**2 / u.s**2, u.km**2 / u.s**2),
        "formation_time": (u.dimensionless_unscaled, u.dimensionless_unscaled),
        "bhmass": (1e10 * u.M_sun / h, u.M_sun),
        "bhmdot": (1e10 * u.M_sun / (h * u.kpc / (u.km / u.s)), u.M_sun / u.yr),
    }

    if dataset not in conversions:
        logger.warning(f"{dataset} not found in GIZMO conversions list.")
        return 1.0

    snap_unit, internal_unit = conversions[dataset]

    return (1 * snap_unit).to(internal_unit).value  # 1 * snap_unit is needed to convert to an astropy Quantity


class SnapshotReader:
    """
    Base reader class (for inheritance). I tucked it away here.
    """

    def read_header(self, snapshot_path: Path) -> SimulationAttributes:
        """
        Read header attributes and, where necessary, convert units.
        """
        raise NotImplementedError

    def read_dataset(self, snapshot_path: Path, ptype: str, dataset: str) -> np.ndarray:
        """
        Returns array in Octavian code units with the correct dtype.
        """
        raise NotImplementedError

    def available_ptypes(self, snapshot_path: Path) -> list[str]:
        """
        List of available ptypes in Octavian convention (gas, star, etc.)
        """
        raise NotImplementedError

    def read_halo_ids(self, ptype: str) -> np.ndarray:
        """
        Reads snapshot-assigned HaloIDs and maps them to a continuous 0-indexed array with a sentinel value of -1.
        """
        raise NotImplementedError
