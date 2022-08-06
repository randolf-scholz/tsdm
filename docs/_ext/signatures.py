r"""Extension for signatures.

Essentially just an admonition box with no body, only a title.
"""

from docutils import nodes
from docutils.parsers.rst import Directive


class Signature(Directive):
    """Essentially just an admonition box with no body, only a title.

    References
    ----------
    - https://docutils.sourceforge.io/docutils/parsers/rst/directives/admonitions.py

    """

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}
    has_content = True

    node_class = nodes.admonition
    """Subclasses must set this to the appropriate admonition node class."""

    def run(self):
        # Raise an error if the directive does not have contents.
        self.assert_has_content()

        # if not len(self.content) == 1:
        #     raise ValueError("Signature directive must have exactly one argument.")

        text = "\n".join(self.content)
        admonition_node = self.node_class(text, **self.options)
        self.add_name(admonition_node)

        title_text = self.content[0]
        signature_text = f"Signature: {title_text}"

        print(f"{self.lineno=}")

        textnodes, messages = self.state.inline_text(signature_text, self.lineno)
        title = nodes.title(signature_text, "", *textnodes)
        title.source, title.line = self.state_machine.get_source_and_line(self.lineno)
        admonition_node += title
        admonition_node += messages
        if "classes" not in self.options:
            admonition_node["classes"] += ["admonition-" + nodes.make_id("signature")]

        return [admonition_node]

    def __run(self):
        self.assert_has_content()
        text = "\n".join(self.content)
        admonition_node = self.node_class(text, **self.options)
        self.add_name(admonition_node)

        print(self.content)
        print(self.content_offset)
        print(self.arguments)
        print(self.lineno)

        title_text = self.content[0]
        textnodes, messages = self.state.inline_text(title_text, self.lineno)
        title = nodes.title(title_text, "", *textnodes)
        title.source, title.line = self.state_machine.get_source_and_line(self.lineno)
        admonition_node += title
        admonition_node += messages
        if "classes" not in self.options:
            admonition_node["classes"] += ["admonition-" + nodes.make_id(title_text)]
        self.state.nested_parse(self.content, self.content_offset, admonition_node)
        return [admonition_node]


def setup(app):
    app.add_directive("sig", Signature)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
