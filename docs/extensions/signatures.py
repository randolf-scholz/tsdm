r"""Extension for signatures.

Essentially just an admonition box with no body, only a title.
"""

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx


class Signature(Directive):
    r"""Essentially just an admonition box without body, only a title.

    References:
        - https://docutils.sourceforge.io/docutils/parsers/rst/directives/admonitions.py
    """

    required_arguments = 1
    r"""The number of required arguments (default: 0)."""
    optional_arguments = 0
    r"""The number of optional arguments (default: 0)."""
    final_argument_whitespace = True
    r"""A boolean, indicating if the final argument may contain whitespace."""
    has_content = False
    r"""A boolean; True if content is allowed.  Client code must handle the case where
    content is required but not supplied (an empty content list will be supplied)."""

    def run(self) -> list[nodes.Node]:
        # Create a preformatted (literal) node for the signature
        signature_text = self.arguments[0]
        # preformatted_node = nodes.literal(signature_text, signature_text)
        text_node = nodes.Text(signature_text)

        # Create a title node and add the preformatted node to it
        title_node = nodes.title("", "Signature: ", text_node)

        # Create the admonition node
        admonition_node = nodes.admonition()
        admonition_node["classes"] += ["signature"]
        admonition_node += title_node

        return [admonition_node]


def setup(app: Sphinx) -> dict:
    r"""Install the extension."""
    app.add_directive("signature", Signature)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
