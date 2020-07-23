# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import sys
import os
import sequd
sys.path.insert(0, os.path.abspath('../../sequd/'))
sys.path.insert(0, os.path.abspath('../_ext'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

source_suffix = '.rst'
master_doc = 'index'
project = u'sequd'

__version__ = sequd.__version__
__author__ = u'Zebin Yang and Aijun Zhang' 
copyright = '2019, Zebin Yang and Aijun Zhang'
exclude_patterns = ['_build']
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
autoclass_content = "both"

html_show_sourcelink = True
html_context = {
  'display_github': True,
  'github_user': 'ZebinYang',
  'github_repo': 'sequd',
  'github_version': 'master/docs/source/'
}
