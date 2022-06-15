# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

# -- Project information -----------------------------------------------------

project = 'IDG'
copyright = '2021, Bram Veenboer, Sebastiaan van der Tol'
author = 'Bram Veenboer, Sebastiaan van der Tol'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [ "sphinx_rtd_theme", "sphinx.ext.autodoc", "breathe", 'myst_parser' ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_logo = 'idg-logo.svg'

html_css_files = ['extra.css']

html_static_path = ['_static']

# Breathe Configuration
# When using CMake, the 'doc' target already sets breathe_projects.
if 'READTHEDOCS' in os.environ:
    breathe_projects = { "IDG": "../build/doc/doxygen/xml" }

breathe_default_project = "IDG"
