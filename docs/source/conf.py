# Sphinx configuration for DFT Documentation
# Best-practice setup: Read the Docs theme, standard extensions, proper meta

import datetime

# -- Project information -----------------------------------------------------
project = 'Droplet-Film Model Development'
copyright = f'{datetime.datetime.now().year}, Droplet-Film Model Development Project'
author = 'DFT Development Team'
release = '2.0'
version = '2.0'
language = 'en'

# -- General configuration ----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # Google/NumPy style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',       # for :math: in docs
    'sphinx_copybutton',        # copy button for code blocks
]
templates_path = ['_templates']
exclude_patterns = []
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
smartquotes = True

# Napoleon (docstring) settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx (link to Python/NumPy docs)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -- HTML output (Read the Docs theme) ----------------------------------------
html_theme = 'sphinx_rtd_theme'
html_title = 'DFT Documentation'
html_short_title = 'DFT'
html_logo = None
html_favicon = None
html_static_path = []  # add '_static' and put CSS there if customizing
html_last_updated_fmt = '%Y-%m-%d'
html_show_sphinx = False
html_show_copyright = True

html_theme_options = {
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
}

# -- Options for todo extension -----------------------------------------------
todo_include_todos = True
