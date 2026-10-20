# Sphinx configuration for the ColRadPy documentation.
#
# Build from the repository root with
#     sphinx-build -b html docs/source docs/build/html
# or from docs/ with "make html". Read the Docs reads this file through
# .readthedocs.yaml.

import os
import sys

# colradpy from this checkout, and the local extensions
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('_ext'))

# no GUI backend for matplotlib, which colradpy imports
os.environ.setdefault('MPLBACKEND', 'Agg')

project = 'ColRadPy'
copyright = '2018, Curtis Johnson'
author = 'Curtis Johnson'
release = '1.1'
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'hidden_code_block',
]

# energy_balance has numpy style docstrings; the others use :param: fields, which
# napoleon's Google style parser would mangle after 'Args:'
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# nist_read imports pymysql, which is only needed to query the NIST database
autodoc_mock_imports = ['pymysql']
autodoc_member_order = 'bysource'

exclude_patterns = ['theory/scripts', 'theory/figures']
pygments_style = 'sphinx'

html_theme = 'sphinx_rtd_theme'
html_logo = '../../logo.jpg'
html_theme_options = {
    'navigation_depth': 3,
}
