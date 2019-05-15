# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import sys
import os
sys.path.insert(0, os.path.abspath('../../seqmml/'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary'
]
source_suffix = '.rst'
master_doc = 'index'
project = u'SeqMML'
author = u'Zebin Yang and Aijun Zhang' 
exclude_patterns = ['_build']
pygments_style = 'sphinx'
html_theme = 'sphinx_rtd_theme'
autoclass_content = "both"


