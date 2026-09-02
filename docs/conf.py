# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # can't do this one with pathlib

project = "octavius"
copyright = "2026, JP Duminy"
author = "JP Duminy"
release = "0.9.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
]
myst_enable_extensions = ["dollarmath", "colon_fence"]
autodoc_member_order = "bysource"
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
language = "en"
nitpicky = True  # auto-flags link rot

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "octavius"
html_theme_options = {
    "dark_css_variables": {},
    "sidebar_hide_name": False,
    "light_logo": "logo.webp",
    "dark_logo": "logo.webp",
}
html_css_files = ["custom.css"]
html_static_path = ["_static"]

# prevent documentation warnings from unrecognised libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

nitpick_ignore = [
    ("py:class", "fsps.StellarPopulation"),
    ("py:class", "h5py._hl.group.Group"),
    ("py:class", "dict[str"),
    ("py:class", "SnapshotReader"),
    ("py:class", "octavius.data_management.pipeline_management.Internals"),
    ("py:class", "np.ndarray"),
]
