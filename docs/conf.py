#!/usr/bin/env python
r"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
# -- Path setup --------------------------------------------------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------------------------------------------------

master_doc = "index"
project = "tsdm"
project_copyright = "%(year)s, %(author)s" % {
    "year": datetime.date.today().year,
    "author": "Randolf Scholz",
}
author = "Randolf Scholz"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_math_dollar",
]

intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "xarray": ("http://xarray.pydata.org/en/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# -- Options for HTML output -------------------------------------------------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# html_style = "css/my_theme.css"

# -- mathjax options ---------------------------------------------------------------------------------------------------

# mathjax_path = r"https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
# The path to the JavaScript file to include in the HTML files in order to load MathJax.
# The default is the https:// URL that loads the JS files from the jsdelivr Content Delivery Network.
# See the MathJax Getting Started page for details. If you want MathJax to be available offline or without including
# resources from a third-party site, you have to download it and set this value to a different path.

# -- autosummary options -----------------------------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

autosummary_context = {}
# A dictionary of values to pass into the template engine’s context for autosummary stubs files.
autosummary_generate = True
# Boolean indicating whether to scan all found documents for autosummary directives,
# and to generate stub pages for each. It is enabled by default.
autosummary_generate_overwrite = True
# If true, autosummary overwrites existing files by generated stub pages. Defaults to true (enabled).
autosummary_mock_imports = []
# This value contains a list of modules to be mocked up. See autodoc_mock_imports for more details.
# It defaults to autodoc_mock_imports.
autosummary_imported_members = False
# A boolean flag indicating whether to document classes and functions imported in modules. Default is False
autosummary_filename_map = {}
# A dict mapping object names to filenames. This is necessary to avoid filename conflicts where multiple objects
# have names that are indistinguishable when case is ignored, on file systems where filenames are case-insensitive.

# -- autodoc options ---------------------------------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#directive-autoclass

autoclass_content = "class"
# This value selects what content will be inserted into the main body of an autoclass directive.
# The possible values are: (default="class")
# "class"
# Only the class’ docstring is inserted. This is the default.
# You can still document __init__ as a separate method using automethod or the members option to autoclass.
# "both"
# Both the class’ and the __init__ method’s docstring are concatenated and inserted.
# "init"
# Only the __init__ method’s docstring is inserted.
# If the class has no __init__ method or if the __init__ method’s docstring is empty,
# but the class has a __new__ method’s docstring, it is used instead.
autodoc_class_signature = "mixed"
# This value selects how the signautre will be displayed for the class defined by autoclass directive.
# The possible values are: (default="mixed")
# "mixed"
# Display the signature with the class name.
# "separated"
# Display the signature as a method.
autodoc_member_order = "alphabetical"
# This value selects if automatically documented members are sorted alphabetical (value 'alphabetical'),
# by member type (value 'groupwise') or by source order (value 'bysource'). The default is alphabetical.
# Note that for source order, the module must be a Python module with the source code available.
autodoc_default_optioss = {}
# The default options for autodoc directives. They are applied to all autodoc directives automatically.
# It must be a dictionary which maps option names to the values. For example:
#
#     autodoc_default_options = {
#         'members': 'var1, var2',
#         'member-order': 'bysource',
#         'special-members': '__init__',
#         'undoc-members': True,
#         'exclude-members': '__weakref__'
#     }
# Setting None or True to the value is equivalent to giving only the option name to the directives.
# The supported options are 'members', 'member-order', 'undoc-members', 'private-members', 'special-members',
# 'inherited-members', 'show-inheritance', 'ignore-module-all', 'imported-members', 'exclude-members' and
# 'class-doc-from'.
autodoc_docstring_signature = True
# Functions imported from C modules cannot be introspected, and therefore the signature for such functions cannot be
# automatically determined. However, it is an often-used convention to put the signature into the first line of the
# function’s docstring.
# If this boolean value is set to True (which is the default), autodoc will look at the first line of the docstring for
# functions and methods, and if it looks like a signature, use the line as the signature and remove it from the
# docstring content.
# autodoc will continue to look for multiple signature lines, stopping at the first line that does not look like a
# signature. This is useful for declaring overloaded function signatures.
autodoc_mock_imports = []
# This value contains a list of modules to be mocked up.
# This is useful when some external dependencies are not met at build time and break the building process.
# You may only specify the root package of the dependencies themselves and omit the sub-modules:
autodoc_typehints = "none"
# This value controls how to represent typehints. The setting takes the following values:
# 'signature' – Show typehints in the signature (default)
# 'description' – Show typehints as content of the function or method The typehints of overloaded
#                 functions or methods will still be represented in the signature.
# 'none' – Do not show typehints
# 'both' – Show typehints in the signature and as content of the function or method
# Overloaded functions or methods will not have typehints included in the description
# because it is impossible to accurately represent all possible overloads as a list of parameters.
autodoc_typehints_description_target = "all"
# This value controls whether the types of undocumented parameters and return values are
# documented when autodoc_typehints is set to description. The default value is "all", meaning that
# types are documented for all parameters and return values, whether they are documented or not.
# When set to "documented", types will only be documented for a parameter or a return value that is
# already documented by the docstring.
autodoc_type_aliases = {}
# A dictionary for users defined type aliases that maps a type name to the full-qualified object name.
# It is used to keep type aliases not evaluated in the document. Defaults to empty ({}).
# The type aliases are only available if your program enables Postponed Evaluation of Annotations (PEP 563)
# feature via from __future__ import annotations.
autodoc_preserve_defaults = False
# If True, the default argument values of functions will be not evaluated on generating document.
# It preserves them as is in the source code.
autodoc_warningiserror = True
# This value controls the behavior of sphinx-build -W during importing modules. If False is given,
# autodoc forcedly suppresses the error if the imported module emits warnings. By default, True.
autodoc_inherit_docstrings = True
# This value controls the docstrings inheritance. If set to True the docstring for classes or methods,
# if not explicitly set, is inherited from parents. The default is True.


# --  sphinx.ext.napoleon configuration --------------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = True
# True to parse Google style docstrings.
# False to disable support for Google style docstrings.
# Defaults to True.
napoleon_numpy_docstring = True
# True to parse NumPy style docstrings.
# False to disable support for NumPy style docstrings.
# Defaults to True.
napoleon_include_init_with_doc = False
# True to list __init___ docstrings separately from the class docstring.
# False to fall back to Sphinx’s default behavior,
# which considers the __init___ docstring as part of the class documentation.
# Defaults to False.
napoleon_include_private_with_doc = False
# True to include private members (like _membername) with docstrings in the documentation.
# False to fall back to Sphinx’s default behavior.
# Defaults to False.
napoleon_include_special_with_doc = True
# True to include special members (like __membername__) with docstrings in the documentation.
# False to fall back to Sphinx’s default behavior.
# Defaults to True.
napoleon_use_admonition_for_examples = False
# True to use the .. admonition:: directive for the Example and Examples sections.
# False to use the .. rubric:: directive instead.
# One may look better than the other depending on what HTML theme is used.
# Defaults to False.
napoleon_use_admonition_for_notes = False
# True to use the .. admonition:: directive for Notes sections.
# False to use the .. rubric:: directive instead.
# Defaults to False.
napoleon_use_ivar = True
# True to use the :ivar: role for instance variables.
# False to use the .. attribute:: directive instead.
# Defaults to False.
napoleon_use_param = True
# True to use a :param: role for each function parameter.
# False to use a single :parameters: role for all the parameters.
# Defaults to True.
napoleon_use_keyword = True
# True to use a :keyword: role for each function keyword argument.
# False to use a single :keyword arguments: role for all the keywords.
# Defaults to True.
napoleon_use_rtype = False
# True to use the :rtype: role for the return type.
# False to output the return type inline with the description.
# Defaults to True.
napoleon_preprocess_types = True
# True to convert the type definitions in the docstrings as references.
# Defaults to True.
napoleon_type_aliases = {
    # python
    "Callable": "~typing.Callable",
    "Path": "~pathlib.Path",
    "Mapping": "~collections.abc.Mapping",
    # torch
    "Tensor": "~torch.Tensor",
    "nn.Module": "~torch.nn.Module",
    # numpy
    "ArrayLike": "~numpy.typing.ArrayLike",
    "datetime64": "~numpy.datetime64",
    "timedelta64": "~numpy.timedelta64",
    "integer": "~numpy.integer",
    "floating": "~numpy.floating",
    # pandas
    "NA": "~pandas.NA",
    "NaT": "~pandas.NaT",
    "DataFrame": "~pandas.DataFrame",
    "Series": "~pandas.Series",
    "Index": "~pandas.Index",
    "MultiIndex": "~pandas.MultiIndex",
    "CategoricalIndex": "~pandas.CategoricalIndex",
    "TimedeltaIndex": "~pandas.TimedeltaIndex",
    "DatetimeIndex": "~pandas.DatetimeIndex",
    "Categorical": "~pandas.Categorical",
    # xarray
    "DataArray": "~xarray.DataArray",
    "Dataset": "~xarray.Dataset",
    "Variable": "~xarray.Variable",
}

# A mapping to translate type names to other names or references. Works only when napoleon_use_param = True.
# Defaults to None.
napoleon_attr_annotations = True
# True to allow using PEP 526 attributes annotations in classes. If an attribute is documented in the docstring without
# a type and has an annotation in the class body, that type is used.
napoleon_custom_sections = [("Hyperparameters", "params_style")]
# Add a list of custom sections to include, expanding the list of parsed sections. Defaults to None.

# -- end of configuration ----------------------------------------------------------------------------------------------
