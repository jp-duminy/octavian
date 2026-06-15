"""

Boilerplate dictionaries which the testing suite must use to verify NaNs are not showing up in the wrong places.

"""

# NaN here means something is broken
NEVER_NAN = [
        "pos", "vel", "minpotpos", "minpotvel",
        "dicts/masses.total", "dicts/masses.total_30kpc",
        "dicts/velocity_dispersions.total",
        "dicts/radii.total_r20", "dicts/radii.total_half_mass", "dicts/radii.total_r80",
        "dicts/rotation.total_L",
        "dicts/rotation.total_ALPHA", "dicts/rotation.total_BETA",
        "dicts/rotation.total_BoverT", "dicts/rotation.total_kappa_rot",
        "dicts/local_mass_density.300", "dicts/local_mass_density.1000", "dicts/local_mass_density.3000",
        "dicts/local_number_density.300", "dicts/local_number_density.1000", "dicts/local_number_density.3000",
    ]

# NaN here should only occur if the corresponding list length is 0
CONDITIONAL_NAN = {
    "halo_data": [
        ("glist_lengths", [
            "dicts/radii.gas_r20", "dicts/radii.gas_half_mass", "dicts/radii.gas_r80",
            "dicts/velocity_dispersions.gas",
            "dicts/rotation.gas_L",
            "dicts/rotation.gas_ALPHA", "dicts/rotation.gas_BETA",
            "dicts/rotation.gas_BoverT", "dicts/rotation.gas_kappa_rot",
            "dicts/metallicities.mass_weighted",
            "dicts/temperatures.mass_weighted",
        ]),
        ("slist_lengths", [
            "dicts/radii.stellar_r20", "dicts/radii.stellar_half_mass", "dicts/radii.stellar_r80",
            "dicts/velocity_dispersions.stellar",
            "dicts/rotation.stellar_L",
            "dicts/rotation.stellar_ALPHA", "dicts/rotation.stellar_BETA",
            "dicts/rotation.stellar_BoverT", "dicts/rotation.stellar_kappa_rot",
            "dicts/metallicities.stellar",
            "dicts/ages.mass_weighted",
        ]),
        ("dmlist_lengths", [
            "dicts/radii.dm_r20", "dicts/radii.dm_half_mass", "dicts/radii.dm_r80",
            "dicts/velocity_dispersions.dm",
            "dicts/rotation.dm_L",
            "dicts/rotation.dm_ALPHA", "dicts/rotation.dm_BETA",
            "dicts/rotation.dm_BoverT", "dicts/rotation.dm_kappa_rot",
        ]),
        ("bhlist_lengths", [
            "dicts/radii.bh_r20", "dicts/radii.bh_half_mass", "dicts/radii.bh_r80",
            "dicts/velocity_dispersions.bh",
            "dicts/rotation.bh_L",
            "dicts/rotation.bh_ALPHA", "dicts/rotation.bh_BETA",
            "dicts/rotation.bh_BoverT", "dicts/rotation.bh_kappa_rot",
        ]),
    ],
    "galaxy_data": [
        ("glist_lengths", [
            "dicts/radii.gas_r20", "dicts/radii.gas_half_mass", "dicts/radii.gas_r80",
            "dicts/velocity_dispersions.gas",
            "dicts/rotation.gas_L",
            "dicts/rotation.gas_ALPHA", "dicts/rotation.gas_BETA",
            "dicts/rotation.gas_BoverT", "dicts/rotation.gas_kappa_rot",
            "dicts/metallicities.mass_weighted",
            "dicts/temperatures.mass_weighted",
        ]),
        ("slist_lengths", [
            "dicts/radii.stellar_r20", "dicts/radii.stellar_half_mass", "dicts/radii.stellar_r80",
            "dicts/velocity_dispersions.stellar",
            "dicts/rotation.stellar_L",
            "dicts/rotation.stellar_ALPHA", "dicts/rotation.stellar_BETA",
            "dicts/rotation.stellar_BoverT", "dicts/rotation.stellar_kappa_rot",
            "dicts/metallicities.stellar",
            "dicts/ages.mass_weighted",
        ]),
        ("bhlist_lengths", [
            "dicts/radii.bh_r20", "dicts/radii.bh_half_mass", "dicts/radii.bh_r80",
            "dicts/velocity_dispersions.bh",
            "dicts/rotation.bh_L",
            "dicts/rotation.bh_ALPHA", "dicts/rotation.bh_BETA",
            "dicts/rotation.bh_BoverT", "dicts/rotation.bh_kappa_rot",
        ]),
    ],
}

# if empty, should be 0 rather than NaN (e.g. total mass)
ZERO_WHEN_EMPTY = {
    "halo_data": [
        ("glist_lengths", ["dicts/masses.gas", "dicts/masses.HI", "dicts/masses.H2"]),
        ("slist_lengths", ["dicts/masses.stellar"]),
        ("dmlist_lengths", ["dicts/masses.dm"]),
        ("bhlist_lengths", ["dicts/masses.bh"]),
    ],
    "galaxy_data": [
        ("glist_lengths", ["dicts/masses.gas", "dicts/masses.HI", "dicts/masses.H2"]),
        ("slist_lengths", ["dicts/masses.stellar"]),
        ("bhlist_lengths", ["dicts/masses.bh"]),
    ],
}

# same as above but for baryons
BARYON_CONDITIONAL_NAN = [
    "dicts/radii.baryon_r20", "dicts/radii.baryon_half_mass", "dicts/radii.baryon_r80",
    "dicts/velocity_dispersions.baryon",
    "dicts/rotation.baryon_L",
    "dicts/rotation.baryon_ALPHA", "dicts/rotation.baryon_BETA",
    "dicts/rotation.baryon_BoverT", "dicts/rotation.baryon_kappa_rot",
]

# NaN can occur for zero division, make a note
SOFT_NAN = [
    "dicts/metallicities.sfr_weighted",
    "dicts/metallicities.mass_weighted_cgm", "dicts/metallicities.temp_weighted_cgm",
    "dicts/temperatures.mass_weighted_cgm", "dicts/temperatures.metal_weighted_cgm",
    "dicts/ages.metal_weighted",
    "dicts/virial_quantities.r200", "dicts/virial_quantities.circular_velocity",
    "dicts/virial_quantities.spin_param", "dicts/virial_quantities.temperature",
    "dicts/virial_quantities.r200c", "dicts/virial_quantities.r500c", "dicts/virial_quantities.r2500c",
    "dicts/virial_quantities.m200c", "dicts/virial_quantities.m500c", "dicts/virial_quantities.m2500c",
]