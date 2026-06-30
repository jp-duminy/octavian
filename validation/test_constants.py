"""

Boilerplate dictionaries which the testing suite must use to verify NaNs are not showing up in the wrong places.

"""

# nan here means something is broken
NEVER_NAN = {
    "halo_data": [
        "properties/core/mass_total",
        "properties/core/velocity_dispersion_total",
        "properties/core/radius_r20_total",
        "properties/core/radius_half_mass_total",
        "properties/core/radius_r80_total",
        "properties/core/x_total", "properties/core/y_total", "properties/core/z_total",
        "properties/core/vx_total", "properties/core/vy_total", "properties/core/vz_total",
        "properties/core/minpot_x", "properties/core/minpot_y", "properties/core/minpot_z",
        "properties/core/minpot_vx", "properties/core/minpot_vy", "properties/core/minpot_vz",
    ],
    "galaxy_data": [
        "properties/core/mass_baryon",
        "properties/core/velocity_dispersion_baryon",
        "properties/core/radius_r20_baryon",
        "properties/core/radius_half_mass_baryon",
        "properties/core/radius_r80_baryon",
        "properties/core/x_baryon", "properties/core/y_baryon", "properties/core/z_baryon",
        "properties/core/vx_baryon", "properties/core/vy_baryon", "properties/core/vz_baryon",
        "properties/core/BoverT_baryon", "properties/core/kappa_rot_baryon",
        "properties/environment/local_mass_density_300kpc",
        "properties/environment/local_mass_density_1000kpc",
        "properties/environment/local_mass_density_3000kpc",
        "properties/environment/local_number_density_300kpc",
        "properties/environment/local_number_density_1000kpc",
        "properties/environment/local_number_density_3000kpc",
        "properties/environment/mass_baryon_30kpc",
    ],
}

# nan here should only occur if the group is empty
CONDITIONAL_NAN = {
    "halo_data": [
        ("ntotal", [
            "properties/core/ALPHA_total",
            "properties/core/BETA_total",
            "properties/core/Lx_total", "properties/core/Ly_total", "properties/core/Lz_total",
        ]),
        ("nbaryon", [
            "properties/core/radius_r20_baryon", "properties/core/radius_half_mass_baryon", "properties/core/radius_r80_baryon",
            "properties/core/velocity_dispersion_baryon",
            "properties/core/ALPHA_baryon", "properties/core/BETA_baryon",
            "properties/core/Lx_baryon", "properties/core/Ly_baryon", "properties/core/Lz_baryon",
        ]),
        ("glist_lengths", [
            "properties/core/radius_r20_gas", "properties/core/radius_half_mass_gas", "properties/core/radius_r80_gas",
            "properties/core/velocity_dispersion_gas",
            "properties/core/Lx_gas", "properties/core/Ly_gas", "properties/core/Lz_gas", 
            "properties/core/ALPHA_gas", "properties/core/BETA_gas",
            "properties/particle_specific/metallicity_mass_weighted",
            "properties/particle_specific/temp_mass_weighted",
        ]),
        ("slist_lengths", [
            "properties/core/radius_r20_star", "properties/core/radius_half_mass_star", "properties/core/radius_r80_star",
            "properties/core/velocity_dispersion_star",
            "properties/core/Lx_star", "properties/core/Ly_star", "properties/core/Lz_star", 
            "properties/core/ALPHA_star", "properties/core/BETA_star",
            "properties/particle_specific/metallicity_stellar",
            "properties/particle_specific/age_mass_weighted",
        ]),
        ("dmlist_lengths", [
            "properties/core/radius_r20_dm", "properties/core/radius_half_mass_dm", "properties/core/radius_r80_dm",
            "properties/core/velocity_dispersion_dm",
            "properties/core/Lx_dm", "properties/core/Ly_dm", "properties/core/Lz_dm", 
            "properties/core/ALPHA_dm", "properties/core/BETA_dm",
        ]),
        ("bhlist_lengths", [
            "properties/core/radius_r20_bh", "properties/core/radius_half_mass_bh", "properties/core/radius_r80_bh",
            "properties/core/velocity_dispersion_bh",
            "properties/core/Lx_bh", "properties/core/Ly_bh", "properties/core/Lz_bh", 
            "properties/core/ALPHA_bh", "properties/core/BETA_bh",
        ]),
    ],
    "galaxy_data": [
        ("nbaryon", [
            "properties/core/ALPHA_baryon",
            "properties/core/BETA_baryon",
            "properties/core/Lx_baryon", "properties/core/Ly_baryon", "properties/core/Lz_baryon",
            "properties/core/radius_r20_baryon", "properties/core/radius_half_mass_baryon", "properties/core/radius_r80_baryon",
            "properties/core/velocity_dispersion_baryon",
        ]),
        ("glist_lengths", [
            "properties/core/radius_r20_gas", "properties/core/radius_half_mass_gas", "properties/core/radius_r80_gas",
            "properties/core/velocity_dispersion_gas",
            "properties/core/Lx_gas", "properties/core/Ly_gas", "properties/core/Lz_gas", 
            "properties/core/ALPHA_gas", "properties/core/BETA_gas",
            "properties/core/BoverT_gas", "properties/core/kappa_rot_gas",
            "properties/particle_specific/metallicity_mass_weighted",
            "properties/particle_specific/temp_mass_weighted",
        ]),
        ("slist_lengths", [
            "properties/core/radius_r20_star", "properties/core/radius_half_mass_star", "properties/core/radius_r80_star",
            "properties/core/velocity_dispersion_star",
            "properties/core/Lx_star", "properties/core/Ly_star", "properties/core/Lz_star", 
            "properties/core/ALPHA_star", "properties/core/BETA_star",
            "properties/core/BoverT_star", "properties/core/kappa_rot_star",
            "properties/particle_specific/metallicity_stellar",
            "properties/particle_specific/age_mass_weighted",
        ]),
        ("bhlist_lengths", [
            "properties/core/radius_r20_bh", "properties/core/radius_half_mass_bh", "properties/core/radius_r80_bh",
            "properties/core/velocity_dispersion_bh",
            "properties/core/Lx_bh", "properties/core/Ly_bh", "properties/core/Lz_bh", 
            "properties/core/ALPHA_bh", "properties/core/BETA_bh",
            "properties/core/BoverT_bh", "properties/core/kappa_rot_bh",
        ]),
    ],
}

# these can be zero, but if they are zero the corresponding group should be empty
ZERO_WHEN_EMPTY = {
    "halo_data": [
        ("glist_lengths", ["properties/core/mass_gas", "properties/particle_specific/mass_HI", "properties/particle_specific/mass_H2"]),
        ("slist_lengths", ["properties/core/mass_star"]),
        ("dmlist_lengths", ["properties/core/mass_dm"]),
        ("bhlist_lengths", ["properties/core/mass_bh"]),
    ],
    "galaxy_data": [
        ("glist_lengths", ["properties/core/mass_gas", "properties/particle_specific/mass_HI", "properties/particle_specific/mass_H2"]),
        ("slist_lengths", ["properties/core/mass_star"]),
        ("bhlist_lengths", ["properties/core/mass_bh"]),
    ],
}

# these can be nan (zero metallicity/sfr), but if a large percentage is, this is problematic
SOFT_NAN = [
    "properties/particle_specific/metallicity_sfr_weighted",
    "properties/particle_specific/metallicity_mass_weighted_cgm",
    "properties/particle_specific/metallicity_temp_weighted_cgm",
    "properties/particle_specific/temp_mass_weighted_cgm",
    "properties/particle_specific/temp_metal_weighted_cgm",
    "properties/particle_specific/age_metal_weighted",
    "properties/core/r200", "properties/core/circular_velocity",
    "properties/core/spin_param", "properties/core/virial_temperature",
    "properties/core/r200c", "properties/core/r500c", "properties/core/r2500c",
    "properties/core/m200c", "properties/core/m500c", "properties/core/m2500c",
]