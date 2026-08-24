"""

Octavius units/dtype conventions and conversions.
Also contains backend dataclasses.

"""

# type checking (semantic)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parallel_reading import RedistributionMap
    from mpi4py.MPI import Comm

# default packages
from dataclasses import dataclass, field
from yaml import safe_load
from pathlib import Path

# other packages
from astropy.constants import codata2022 as codata
from astropy.constants import iau2015 as iau
import astropy.units as u
from astropy.cosmology import FLRW
import numpy as np

# internal imports
from ..log import get_logger

logger = get_logger()

# for config parsing
CONFIG_FIELDS = frozenset({"thresholds", "physics", "fof6d", "properties", "photometry", "parallelism", "logging"})
FILEPATHS = frozenset({"snapshot_path", "output_dir", "halo_id_filepath", "photometry_table_filepath"})


@dataclass(frozen=True, slots=True)
class OctaviusConfig:
    """
    Octavius config object containing user-configured physics/runtime parameters. This can either be directly parsed from a .yaml file with the class method (recommended), or created directly.

    Methods
    -------
    from_yaml()
        Parses a .yaml file into the config (recommended usage).

    Notes
    -----

    - 'b' controls what fraction of the mean interparticle separation the linking length is. This is usually set to 0.02.
    - 'velocity_factor' is in units of the local velocity dispersion, and controls how many standard deviations from the local velocity dispersion a particle considers its neighbours to be linked in phase space.
    - FRAD is the radiative efficiency (in the accretion formula, usually 0.1).
    - MU is the mean molecular weight (in the virial scaling formulae, usually 0.6)
    - n_chunks does not represent chunking through the full pipeline, rather, chunking on reading datasets. This is to alleviate stress on filesystems from multiple ranks simultaneously requesting multi-GB reads, but in practice leaving this at 10 is fine.
    """

    simulation_type: str

    halo_id_source: str  # filepath is at the bottom

    stages: dict[str, bool]
    n_chunks: int
    process_ptypes: dict[str, bool]

    min_stars_per_galaxy: int
    min_dm_per_halo: int

    nH_lim: float
    T_lim: float
    FRAD: float
    MU: float

    b: float
    velocity_factor: float

    radial_quantiles: dict[str, float]
    aperture_size: list[int]
    virial_factors: list[int]
    density_radii: list[int]

    bands: list[str]
    extinction_law: str
    viewing_axis: str
    use_dust: bool
    use_cosmic_extinction: bool
    interpolation_bins: int
    kernel_type: str
    power_law_alpha: float
    split_age: float

    cores_per_rank: int

    terminal_output_level: str
    keep_logs: bool

    snapshot_path: Path | None = None
    output_dir: Path | None = None
    halo_id_filepath: Path | None = None
    photometry_table_filepath: Path | None = None

    @classmethod
    def from_yaml(cls, config_path: Path) -> OctaviusConfig:
        """
        Parses a config .yaml parameter file into the dataclass used internally. Names of entries themselves and the file layout must not be changed.

        Parameters
        ----------
        config_path: pathlib.Path
            Path object pointing to the config file.

        Returns
        -------
        config: OctaviusConfig
            The config dataclass.
        """
        with open(config_path, "r") as f:
            raw = safe_load(f)

        flat = _flatten_config(raw)

        for key in FILEPATHS:
            if key in flat and flat[key] is not None:
                flat[key] = Path(flat[key]).expanduser()

        return cls(**flat)  # keyword unpacking saves us from a 35 argument instantiation


def _flatten_config(raw: dict) -> dict:
    """
    Flattens the raw config (in dict form, parsed with yaml) from its nested structure into flat fields (obeying old behaviour so you can just key off config).
    """
    flat: dict = {}
    for key, value in raw.items():
        if isinstance(value, dict) and key in CONFIG_FIELDS:
            for inner_key in value:
                if inner_key in flat:
                    raise ValueError(f"{inner_key} is duplicated in the config.")
            flat.update(value)
        else:
            flat[key] = value

    return flat


@dataclass(frozen=True, slots=True)
class OctaviusConstants:
    """
    The internal constants and scaling relations used in the Octavius pipeline.

    Notes
    -----

    - This is backed by the astropy API and uses the CODATA 2022 & IAU 2015 datasets.
    - Contains some scaling factors.

    To get the astropy datasets please run:

    - from astropy.constants import iau2015 as iau
    - from astropy.constants import codata2022 as codata
    - import astropy.units as u
    """

    # config-dependent parameters
    mu: float = 0.6  # (assuming X=0.7, Y=0.28, Z=0.02)
    frad: float = 0.1

    # fundamental values (CODATA/IAU)
    G_CGS: float = codata.G.cgs.value
    C_CGS: float = codata.c.cgs.value
    C_KMS: float = codata.c.to(u.km / u.s).value
    PROTON_MASS_G: float = codata.m_p.cgs.value
    BOLTZMANN_CGS: float = codata.k_B.cgs.value
    SIGMA_T_CGS: float = codata.sigma_T.cgs.value
    M_SUN_G: float = iau.M_sun.cgs.value
    L_SUN_CGS: float = iau.L_sun.cgs.value
    KPC_CM: float = iau.kpc.cgs.value
    PC_CM: float = iau.pc.cgs.value
    KPC_M: float = iau.kpc.si.value
    GYR_S: float = (1 * u.Gyr).to(u.s).value

    X_H: float = 0.76  # primoridal hydrogen fraction
    MW_DUST_TO_METAL: float = 0.4 / 0.6  # dwek 1998, watson 2011
    Z_SUN_WATSON: float = 0.0189  # watson 2011
    Z_SUN_ASPLUND: float = 0.0134  # asplund 2009
    AV_TO_NH: float = 2.2e21  # watson 2011

    # derived unit conversions
    G_VCIRC: float = codata.G.to(u.km**2 * u.kpc / (u.M_sun * u.s**2)).value

    # derived factors
    VIRIAL_TEMP_FACTOR: float = field(init=False)
    EDD_FACTOR: float = field(init=False)
    Z_COL_TO_AV: float = field(init=False)

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
    scale_factor: float
    w_0: float
    w_a: float
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
    "dust_mass": DatasetUnits(unit=(u.M_sun)),  # solar masses
    "smoothing_length": DatasetUnits(unit=u.kpc, a_exponent=1),  # ckpc
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
        "dust_mass": (1e10 * u.M_sun / h),
        "smoothing_length": (u.kpc * a / h),
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
    global_indices: dict[str, np.ndarray] | None = NotImplemented
    particle_counts: dict[str, int] | None = NotImplemented

    def set_maps(
        self,
        slabs: dict[str, slice],
        masks: dict[str, np.ndarray],
        maps: dict[str, RedistributionMap],
        comm: Comm | None,
    ) -> None:
        """
        Sets the per-rank slabs; global particle redistribution map; corresponding halo threshold masks; and  and comm for MPI reading.
        """
        raise NotImplementedError

    def read_header(self) -> SimulationAttributes:
        """
        Read header attributes and, where necessary, convert units.
        """
        raise NotImplementedError

    def has_dataset(self, ptype: str, dataset: str) -> bool:
        """
        Checks whether a dataset exists in the snapshot.
        """
        raise NotImplementedError

    def read_dataset(self, ptype: str, dataset: str) -> np.ndarray:
        """
        Returns array in Octavius code units with the correct dtype.
        """
        raise NotImplementedError

    def available_ptypes(self) -> list[str]:
        """
        List of available ptypes in Octavius convention (gas, star, etc.)
        """
        raise NotImplementedError

    def read_halo_ids(self, ptype: str, slab: slice = slice(None)) -> np.ndarray:
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


def output_catalogue_path(snapshot_path: Path, output_dir: Path) -> Path:
    """
    Returns the Path object pointing to the final production output analysis catalogue filename.
    """
    return output_dir / f"octavius_{snapshot_path.stem}.hdf5"
