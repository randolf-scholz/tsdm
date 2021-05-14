# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import datetime


# -- Project information -----------------------------------------------------

master_doc = "index"
project = 'tsdm'
copyright = '%(year)s, %(author)s' % dict(
    year=datetime.date.today().year,
    author='Randolf Scholz'
)
author = 'Randolf Scholz'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    # 'sphinx_autodoc_typehints',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_math_dollar',
]

mathjax_config = {
    'tex2jax': {
        'inlineMath': [["\\(", "\\)"]],
        'displayMath': [["\\[", "\\]"]],
    },
}

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy"     : ("https://numpy.org/doc/stable/", None),
    "python"    : ("https://docs.python.org/3/", None),
    "scipy"     : ("https://docs.scipy.org/doc/scipy/reference/", None),
    'pandas'    : ('https://pandas.pydata.org/docs', None),
    'torch'     : ('https://pytorch.org/docs/stable/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_typehints = 'description'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# -- sphinx_autodoc_typehints configuration ----------------------------------

# set_type_checking_flag (default: False): if True, set typing.TYPE_CHECKING to True to enable "expensive" typing
# imports
set_type_checking_flag = False
# typehints_fully_qualified (default: False): if True, class names are always fully qualified (e.g.
# module.for.Class). If False, just the class name displays (e.g. Class)
typehints_fully_qualified = False
# always_document_param_types (default: False): If False, do not add type info for undocumented parameters. If True,
# add stub documentation for undocumented parameters to be able to add type info.
always_document_param_types = False
# typehints_document_rtype (default: True): If False, never add an :rtype: directive. If True, add the :rtype:
# directive if no existing :rtype: is found.
typehints_document_rtype = False
# simplify_optional_unions (default: True): If True, optional parameters of type "Union[...]" are simplified as being
# of type Union[..., None] in the resulting documention (e.g. Optional[Union[A, B]] -> Union[A, B, None]). If False,
# the "Optional"-type is kept. Note: If False, any Union containing None will be displayed as Optional! Note: If an
# optional parameter has only a single type (e.g Optional[A] or Union[A, None]), it will always be displayed as
# Optional!
simplify_optional_unions = True
