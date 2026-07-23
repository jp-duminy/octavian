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

    simulation_type: str

    halo_id_source: str  # filepath is at the bottom

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
    keep_logs: bool

    halo_id_filepath: Path | None = None  # because of Python syntax this has to go at the bottom (it has default arg)

    @classmethod
    def from_yaml(cls, config_path: Path) -> OctavianConfig:
        """
        Parses a config.yaml file into the dataclass.
        """
        with open(config_path, "r") as f:
            raw = safe_load(f)

        return cls(
            simulation_type=raw["simulation_type"],
            halo_id_source=raw["halo_id_source"],
            halo_id_filepath=Path(raw["halo_id_filepath"]).expanduser() if "halo_id_filepath" in raw else None,
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
            keep_logs=raw["keep_logs"],
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


@dataclass(frozen=True, slots=True)
class DatasetUnits:
    """
    Astropy unit with the a_exponent baked in; SWIFT cares not for h factors and we divide them out in GIZMO, so we omit it.
    """

    unit: u.UnitBase
    a_exponent: int = 0


CODE_UNITS = {
    "pos": DatasetUnits(unit=u.kpc, a_exponent=1),  # kpc * a
    "vel": DatasetUnits(unit=(u.km / u.s)),  # peculiar
    "potential": DatasetUnits(unit=u.km**2 / u.s**2),  # (km/s)**2
    "mass": DatasetUnits(unit=u.M_sun),  # solar masses
    "internal_energy": DatasetUnits(unit=(u.cm**2 / u.s**2)),  # CGS
    "temperature": DatasetUnits(unit=u.K),  # kelvin
    "metallicity": DatasetUnits(unit=u.dimensionless_unscaled),  # dimensionless
    "age": DatasetUnits(unit=u.Gyr),
    "rho": DatasetUnits(unit=(u.g / u.cm**3)),  # CGS
    "rhocrit": DatasetUnits(unit=(u.M_sun / u.kpc**3)),
    "helium_fraction": DatasetUnits(unit=u.dimensionless_unscaled),
    "electron_abundance": DatasetUnits(unit=u.dimensionless_unscaled),
    "fHI": DatasetUnits(unit=u.dimensionless_unscaled),
    "fH2": DatasetUnits(unit=u.dimensionless_unscaled),
    "sfr": DatasetUnits(unit=(u.M_sun / u.yr)),  # solar masses/yr
    "bhmass": DatasetUnits(unit=u.M_sun),  # solar masses
    "bhmdot": DatasetUnits(unit=(u.M_sun / u.yr)),  # solar masses/yr
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
    "age": np.float64,
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
        "pos": (u.kpc * a / h),
        "vel": (u.km / u.s * np.sqrt(a)),
        "potential": (u.km**2 / u.s**2),
        "mass": (1e10 * u.M_sun / h),  # 1e10 factor is just a Gizmo scaling convention
        "rho": (1e10 * u.M_sun * h**2 / (u.kpc**3 * a**3)),
        "internal_energy": (u.km**2 / u.s**2),
        "sfr": (u.M_sun / u.yr),
        "metallicity": (u.dimensionless_unscaled),
        "fHI": (u.dimensionless_unscaled),
        "fH2": (u.dimensionless_unscaled),
        "age": (u.Gyr),
        "bhmass": (1e10 * u.M_sun / h),
        "bhmdot": (1e10 * u.M_sun / (h * u.kpc / (u.km / u.s))),
        "helium_fraction": (u.dimensionless_unscaled),
        "electron_abundance": (u.dimensionless_unscaled),
    }

    snap_unit = conversions[dataset]
    target = CODE_UNITS[dataset]
    target_quantity = a**target.a_exponent * target.unit

    factor = (
        (1 * snap_unit) / target_quantity
    ).decompose()  # 1 * snap_unit is needed to convert to an astropy Quantity
    assert factor.unit == u.dimensionless_unscaled, f"Unit mismatch for {dataset}"

    return factor.value


class SnapshotReader:
    """
    Base reader class (for inheritance). I tucked it away here.
    """

    inverse_ptype_map: dict[str, str] = NotImplemented
    dataset_map: dict[str, dict[str, str]] = NotImplemented
    simulation_attributes: SimulationAttributes = NotImplemented
    indices: dict[str, np.ndarray] | None = NotImplemented

    def set_indices(self, indices: dict[str, np.ndarray]) -> None:
        """
        Sets the indices (if in parallel) dictionary to the per-rank read masks.
        """
        raise NotImplementedError

    def read_header(self) -> SimulationAttributes:
        """
        Read header attributes and, where necessary, convert units.
        """
        raise NotImplementedError

    def read_dataset(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Returns array in Octavian code units with the correct dtype.
        """
        raise NotImplementedError

    def available_ptypes(self) -> list[str]:
        """
        List of available ptypes in Octavian convention (gas, star, etc.)
        """
        raise NotImplementedError

    def read_halo_ids(self, ptype: str) -> np.ndarray:
        """
        Reads snapshot-assigned HaloIDs and maps them to a continuous 0-indexed array with a sentinel value of -1.
        """
        raise NotImplementedError

    def read_particle_ids(self, ptype: str) -> np.ndarray:
        """
        Reads snapshot-assigned particle IDs for a specified ptype.
        """
        raise NotImplementedError

    def read_temperature(self, ptype: str) -> np.ndarray:
        """
        Temperature is usually computed from multiple datasets, and how it can be computed differs between readers.
        """
        raise NotImplementedError


def intermediate_catalogue_path(directory: Path, rank: int) -> Path:
    """
    Returns the Path object pointing to the intermediate analysis catalogue filename for a rank.
    """
    return directory / f"rank_{rank}_analysis.hdf5"


def output_catalogue_path(snapshot_path: Path, output_dir: Path) -> Path:
    """
    Returns the Path object pointing to the final production output analysis catalogue filename.
    """
    return output_dir / f"octavian_{snapshot_path.stem}.hdf5"
