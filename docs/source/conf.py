import os, re, sys

# Path: two levels up from docs/source/ to repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

project = 'pynalgo'
copyright = '2026, Boris Daszuta'
author = 'Boris Daszuta'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []
language = 'en'
suppress_warnings = ['docutils']

# Autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': False,
}
autodoc_typehints = 'none'

# Napoleon: Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'numba': ('https://numba.readthedocs.io/en/stable/', None),
}

# HTML: sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {}
html_static_path = ['_static']
html_title = 'pynalgo'
html_short_title = 'pynalgo'

# Strip ASCII-art banner headers from module docstrings
_PYNALGO_HEADER_RE = re.compile(
    r'\A\s*,-[*]\s*\n\s*\(_\)\s*\n\s*'
    r'(?:@(?:author|SPDX-License-Identifier):[^\n]*\n\s*)*'
    r'@function:\s*([^\n]*)'
)

def _strip_pynalgo_header(app, what, name, obj, options, lines):
    if what != 'module' or not lines:
        return
    text = '\n'.join(lines)
    m = _PYNALGO_HEADER_RE.match(text)
    if m:
        lines[:] = [m.group(1)]

def setup(app):
    app.connect('autodoc-process-docstring', _strip_pynalgo_header)
