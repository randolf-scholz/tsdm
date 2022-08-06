r"""Example extension for Sphinx.

References
----------
- https://www.sphinx-doc.org/en/master/development/tutorials/helloworld.html
- https://www.sphinx-doc.org/en/master/development/tutorials/todo.html
"""

from docutils import nodes
from docutils.parsers.rst import Directive


class HelloWorld(Directive):
    def run(self):
        paragraph_node = nodes.paragraph(text="Hello World!")
        return [paragraph_node]


def setup(app):
    app.add_directive("helloworld", HelloWorld)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
