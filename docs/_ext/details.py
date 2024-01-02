r"""Extension for details.

Shortcut for sphinx-toggle button admonition.

.. admonition:: Show/Hide
  :class: dropdown

  hidden message
"""

from pprint import pprint

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx


class Details(Directive):
    """Collapsible admonition box.

    .. admonition:: Show/Hide
      :class: dropdown

      hidden message

    References:
        - https://stackoverflow.com/a/74114447/9318372
        - https://sphinx-togglebutton.readthedocs.io/en/latest/
    """

    required_arguments = 0
    r"""The number of required arguments (default: 0)."""
    optional_arguments = 1
    r"""The number of optional arguments (default: 0)."""
    final_argument_whitespace = True
    r"""A boolean, indicating if the final argument may contain whitespace."""
    has_content = True
    r"""A boolean; True if content is allowed.  Client code must handle the case where
    content is required but not supplied (an empty content list will be supplied)."""

    def run(self) -> list[nodes.Node]:
        # Create the admonition node
        admonition_node = nodes.admonition()
        admonition_node["classes"] += ["dropdown"]

        # Set the title of the dropdown
        title_text = self.arguments[0]
        textnodes, messages = self.state.inline_text(title_text, self.lineno)
        title = nodes.title(title_text, "", *textnodes)
        admonition_node += title
        admonition_node += messages

        # Parse the content of the directive
        self.state.nested_parse(self.content, self.content_offset, admonition_node)

        return [admonition_node]


def setup(app: Sphinx) -> dict:
    r"""Install the extension."""
    app.add_directive("details", Details)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
