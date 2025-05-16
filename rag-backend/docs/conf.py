# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from src.app import app as flask_app

# Setup Flask autodoc
autohttp_flask_app = flask_app

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Rag backend'
copyright = '2025, Felix Pigeon'
author = 'Felix Pigeon'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autoapi.extension',
    'sphinxcontrib.httpdomain',
    'sphinxcontrib.autohttp.flask',
    'sphinxcontrib.autohttp.flaskqref',
]

# Add AutoAPI settings
autoapi_type = 'python'
autoapi_dirs = ['../src']
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'imported-members',
]


autoapi_ignore = ['*/baml_client/*']


templates_path = ['_templates']
exclude_patterns = []

language = 'fr'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
