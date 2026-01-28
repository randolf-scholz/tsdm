r"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

Useful links:

- https://docutils.sourceforge.io/docs/user/rst/quickref.html
- https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
- https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
"""
# ruff: line-length=100

import datetime
import os
import sys
from importlib import metadata

# setup path
os.environ["GENERATING_DOCS"] = "true"
sys.path.insert(0, os.path.abspath("../src"))  # Source code dir relative to this file
sys.path.append(os.path.abspath("./extensions"))

AUTHOR = "Randolf Scholz"
MODULE = "tsdm"
MODULE_DIR = "src/tsdm"
VERSION = metadata.version(MODULE)
YEAR = datetime.datetime.now().year

# region General Configuration ---------------------------------------------------------
# SEE: https://www.sphinx-doc.org/en/master/usage/configuration.html

# project tags
project = f"{MODULE}"
author = "Randolf Scholz"
project_copyright = f"{YEAR}, {AUTHOR}"
version = VERSION  # major project version, e.g. '2.6'
release = version  # full project version, e.g. '2.6.0rc1' or '2.6+git@abcdef'

# region General Configuration ---------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Sphinx builtin extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # 1st party extensions
    "details",
    "signatures",
    # 3rd party extensions
    # "autoapi.extension",
    # "myst_parser",
    "sphinx_copybutton",
    "sphinx_math_dollar",
    "sphinx_togglebutton",
    # "sphinx_autodoc_typehints",
    # "sphinx_toolbox.more_autodoc.typehints",
    # "sphinx_toolbox.more_autodoc.typevars",
    # "sphinx_toolbox.more_autodoc.genericalias",
]
needs_extensions = {}  # minimum version requirements for extensions
# manpages_url = ...  # url for cross-referencing manpages
# today = ...  # override today's date
today_fmt = "%Y-%m-%d"  # format for date in the docs

# source file options
root_doc = "index"  # the master toctree document
exclude_patterns = ["_*", ".*"]  # glob-style patterns that should be excluded
include_patterns = ["**"]  # glob-style patterns [1] that are used to find source files.
templates_path = ["_templates"]  # paths that contain the template files

# markup options
rst_epilog = ""  # reStructuredText to append to every document
rst_prolog = ""  # reStructuredText to prepend to every document
show_authors = True  # enable `.. codeauthor::` and `.. sectionauthor::` directives
default_role = "py:obj"  # reStructuredText role use for backtick strings like `object`
primary_domain = "py"  # default domain for objects (e.g. :class:, :func:, :mod:)
keep_warnings = False  # Keep warnings in the rendered documents.

# warning control
suppress_warnings = []  # warning types to suppress arbitrary warning messages.
show_warning_types = True  # show warning type in the output

# highlighting configuration
highlight_language = "default"  # default language for syntax highlighting
pygments_style = "default"  # style name for Pygments highlighting of source code.
pygments_options = {}  # options for Pygments syntax highlighting

# options for object signatures
add_function_parentheses = False  # render :func:`input` as input() instead of input
maximum_signature_line_length = 88  # maximum line length for function signatures
toc_object_entries = True  # include objects in the TOC
toc_object_entries_show_parents = "domain"  # how object TOC entries are displayed

# options for python domain
add_module_names = False  # prepend module names to all object names
modindex_common_prefix = []  # list of prefixes to ignore in object names
python_display_short_literal_types: True  # display Literal["egg"] as "egg"
python_maximum_signature_line_length = 88  # maximum line length for function signatures
python_use_unqualified_type_names = True  # If true, suppress the module name
trim_doctest_flags = True  # remove common whitespace from doctest blocks
# endregion General Configuration ------------------------------------------------------


# region HTML Configuration ------------------------------------------------------------
# SEE: https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme_path = ["theme"]  # paths that contain custom themes
# piccolo_theme, karma_sphinx_theme, sphinx_rtd_theme, pydata_sphinx_theme, sphinx_typo3_theme
html_theme = "pydata_sphinx_theme"  # select the theme
html_theme_options = {
    # faster builds?  SEE: https://stackoverflow.com/a/52175461
    "collapse_navigation": False,
    "navigation_depth": 2,
    #
    "header_links_before_dropdown": 7,
    "icon_links": [
        {
            "name": "GitHub",
            "url": f"https://github.com/randolf-scholz/{MODULE}",
            "icon": "fa-brands fa-github",
        },
        # {
        #     "name": "PyPI",
        #     "url": "https://pypi.org/project/pydata-sphinx-theme",
        #     "icon": "fa-custom fa-pypi",
        # },
    ],
    # "external_links": [
    #     {"url": "https://pydata.org", "name": "PyData"},
    #     {"url": "https://numfocus.org/", "name": "NumFocus"},
    #     {"url": "https://numfocus.org/donate", "name": "Donate to NumFocus"},
    # ],
}

html_style = []  # style sheets to use for HTML pages
html_title = f"{MODULE} {VERSION}"  # <title> tag
html_short_title = MODULE  # shorter title used for links in the header
html_baseurl = ""  # The base URL which points to the root of the HTML documentation.
html_codeblock_linenos_style = "inline"  # style for line numbers in code-blocks
html_context = {}  # options to pass to the template engine
html_logo = ""  # path/url to the project logo
html_favicon = ""  # path/url to the favicon (icon in the browser tab)
html_css_files = []  # A list of CSS files
html_js_files = []  # A list of JavaScript filename.
html_static_path = ["_static"]  # A list of paths that contain custom static files
html_extra_path = []  # extra files not directly related to the documentation
html_permalinks = True  # Add link anchors to sections
html_permalinks_icon = "§"  # A text for permalinks for each heading
html_sidebars = {}  # custom sidebar templates
html_additional_pages = {}  # Additional pages such as error pages
html_domain_indices = True  # Create index pages for the domain indices
html_use_index = True  # Add index to html pages
html_split_index = False  # Split the index into individual pages for each letter
html_copy_source = True  # Copy source files to the output directory
html_show_sourcelink = True  # Show link to the source code
html_show_sourcelink_suffix = ".txt"  # Suffix for source links
html_file_suffix = ".html"  # Suffix for HTML files
html_link_suffix = ".html"  # Suffix for links
html_show_copyright = True  # Show copyright
html_show_sphinx = True  # Show sphinx version
html_show_search_summary = True  # show text around the keyword
html_output_encoding = "utf-8"  # output encoding
html_compact_lists = True  # compact lists
html_secnumber_suffix = ". "  # suffix for section numbers
html_search_language = "en"  # language for search
html_search_options = {}  # options for search
html_search_scorer = ""  # class for search scoring
html_scaled_image_link = True  # scale images to fit the page
html_math_renderer = "mathjax"  # Math renderer to use
# endregion HTML Configuration ---------------------------------------------------------


# region sphinx.ext.autodoc configuration ----------------------------------------------
# FIXME: https://github.com/olgarithms/sphinx-tutorial/issues/14
# FIXME: https://github.com/sphinx-doc/sphinx/issues/4961
# SEE: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration
autoclass_content = "class"  # docstring to insert in classes: "class", "both", "init"
autodoc_class_signature = "separated"  # display class signatures: "separated", "mixed"
autodoc_member_order = "groupwise"  # order "alphabetical", "groupwise", "bysource"
autodoc_default_options = {  # default options for autodoc directives
    # 'members'           : True,
    # 'undoc-members'     : True,
    # 'private-members'   : True,
    # 'special-members'   : True,
    # 'inherited-members' : True,
    # 'imported-members'  : True,
    # 'exclude-members'   : True,
    # 'ignore-module-all' : True,
    # 'member-order'      : True,
    # 'show-inheritance'  : True,
    # 'class-doc-from'    : True,
    # 'no-value'          : True,
}  # fmt: skip
autodoc_docstring_signature = True  # handling function signatures of C-extensions
autodoc_mock_imports = []  # list of modules to mock
autodoc_typehints = "both"  # show typehints: "signature", "description", "none", "both"
autodoc_typehints_description_target = "all"  # "all", "documented"
autodoc_type_aliases = {  # type aliases (requires PEP 563)
    "Protocol": "typing.Protocol",
}
autodoc_typehints_format = "short"  # format of typehints: "short", "fully-qualified"
autodoc_preserve_defaults = True  # whether to NOT eval() default values
autodoc_warningiserror = True  # turn warnings into errors
autodoc_inherit_docstrings = True  # inherit docstrings from parent classes
# endregion sphinx.ext.autodoc configuration -------------------------------------------


# region sphinx-autoapi configuration --------------------------------------------------
# SEE: https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

# configuration options
autoapi_dirs = [f"../{MODULE_DIR}"]  # Paths (relative or absolute) to the source code
autoapi_template_dir = "_templates/autoapi"  # directory containing custom templates
autoapi_type = "python"  # Set the type of files you are documenting.
autoapi_file_patterns = ["*.py", "*.pyi"]  # glob patterns for finding files
autoapi_generate_api_docs = True  # Whether to generate API docs.

# customization options
autoapi_options = [  # SEE: autodoc_default_options
    "members",
    # "undoc-members",
    # "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    # "imported-members",
]
autoapi_ignore = []  # patterns to ignore when finding files
autoapi_root = "autoapi"  # Path to output the generated AutoAPI files into
autoapi_add_toctree_entry = False  # insert the generated docs into the TOC tree
autoapi_member_order = "groupwise"  # SEE: autodoc_member_order
autoapi_python_class_content = "both"  # SEE: autoclass_content
autoapi_python_use_implicit_namespaces = False  # detect implicit namespaces (PEP 420)
autoapi_prepare_jinja_env = None  # A callback after the Jinja environment is created.
autoapi_own_page_level = "function"  # the level objects are rendered on a single page
autoapi_keep_files = True  # Keep the AutoAPI generated files on the filesystem.
# endregion sphinx-autoapi configuration -----------------------------------------------


# region sphinx.ext.autosectionlabel configuration -------------------------------------
# SEE: https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html
autosectionlabel_prefix_document = False  # prefix label with document name
autosectionlabel_maxdepth = None  # how many section levels to label
# endregion sphinx.ext.autosectionlabel configuration ----------------------------------


# region sphinx.ext.autosummary configuration ------------------------------------------
# SEE: https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
autosummary_context = {}  # options passed to the template engine
autosummary_generate = True  # scan all documents for `.. autosummary::` directives
autosummary_generate_overwrite = True  # overwrite generated stub pages
autosummary_mock_imports = []  # list of modules to mock
autosummary_imported_members = False  # document imported members
autosummary_ignore_module_all = False  # ignore module __all__ attribute
autosummary_filename_map = {}  # dict mapping filenames to objects
# endregion sphinx.ext.autosummary configuration ---------------------------------------


# region sphinx.ext.intersphinx configuration ------------------------------------------
# SEE: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {  # targets for cross-referencing
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "polars": ("https://docs.pola.rs/api/python/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "typing": ("https://typing-extensions.readthedocs.io/en/latest/", None),
}
intersphinx_cache_limit = 5  # maximum number of days to cache remote inventories
intersphinx_timeout = 2  # seconds for timeout
intersphinx_disabled_reftypes = ["std:doc"]  # list of disabled cross-reference types
# endregion sphinx.ext.intersphinx configuration ---------------------------------------


# region sphinx.ext.mathjax configuration ----------------------------------------------
# SEE: https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.mathjax
# SEE: https://docs.mathjax.org/en/latest/web/configuration.html

mathjax_config_path = "_static/mathjax_config.js"

# endregion sphinx.ext.mathjax configuration -------------------------------------------


# region sphinx.ext.napoleon configuration ---------------------------------------------
# SEE: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_google_docstring = True  # parse google style docstrings.
napoleon_numpy_docstring = True  # parse NumPy style docstrings.
napoleon_include_init_with_doc = True  # include __init__
napoleon_include_private_with_doc = False  # include _private members
napoleon_include_special_with_doc = True  # include __dunder__ members
napoleon_use_admonition_for_examples = True  # use .. admonition:: for Examples
napoleon_use_admonition_for_notes = True  # use .. admonition:: for Notes
napoleon_use_admonition_for_references = True  # use .. admonition:: for References
napoleon_use_ivar = True  # use :ivar: role for instance variables
napoleon_use_param = True  # use :param: role for each function parameter
napoleon_use_keyword = True  # use :keyword: role for each function keyword argument
napoleon_use_rtype = True  # use :rtype: role for the return type
napoleon_attr_annotations = True  # PEP 526 style annotations
napoleon_custom_sections = [  # list of custom sections to include
    "Test-Metric",
    "Evaluation Protocol",
    "Paper",
    "Results",
    # allow multiple return values
    # SEE: https://github.com/sphinx-doc/sphinx/issues/9119
    ("Returns", "params_style"),
]
napoleon_preprocess_types = True  # convert type definitions to references
napoleon_type_aliases = {}  # type aliases
# endregion sphinx.ext.napoleon configuration ------------------------------------------


# region sphinx.ext.todo configuration -------------------------------------------------
# SEE: https://www.sphinx-doc.org/en/master/usage/extensions/todo.html
todo_include_todos = True  # include `.. todo::` and `.. todolist::` directives
todo_emit_warnings = False  # If True, todo emits a warning.
todo_link_only = False  # link to the todo, rather than the printing its content.
# endregion sphinx.ext.todo configuration ----------------------------------------------


# region sphinx.ext.viewcode configuration ---------------------------------------------
# SEE: https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html
viewcode_follow_imported_members = True  # follow imported members
viewcode_enable_epub = False  # enable in epub builds
viewcode_line_numbers = True  # print inline line numbers
# endregion sphinx.ext.viewcode configuration ------------------------------------------


# region sphinx_math_dollar configuration ----------------------------------------------
# SEE: https://www.sympy.org/sphinx-math-dollar/#configuration
# math_dollar_debug = True
# math_dollar_node_blacklist = NODE_BLACKLIST + (header, pending_xref_condition)

from sphinx.addnodes import pending_xref_condition  # noqa: E402
from sphinx.util.docutils import register_node  # noqa: E402

register_node(pending_xref_condition)
# endregion sphinx_math_dollar configuration -------------------------------------------


# region MyST Configuration ------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "html_admonition",
    "html_image",
]

# endregion MyST Configuration ---------------------------------------------------------


# -- end of configuration --------------------------------------------------------------
