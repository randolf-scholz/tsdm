{{ name | escape | underline}}

.. automodule:: {{ fullname }}
    {% block attributes %}
    {% if attributes %}
    .. rubric:: Module attributes
    .. autosummary::
      :toctree:  {{ name }}
    {% for item in attributes %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}
    .. rubric:: {{ _('Functions') }}
    .. autosummary::
      :toctree:  {{ name }}
      :nosignatures:
    {% for item in functions %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block classes %}
    {% if classes %}
    .. rubric:: {{ _('Classes') }}
    .. autosummary::
      :toctree:  {{ name }}
      :template: custom-class-template.rst
      :nosignatures:
    {% for item in classes %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block exceptions %}
    {% if exceptions %}
    .. rubric:: {{ _('Exceptions') }}
    .. autosummary::
      :toctree:  {{ name }}
    {% for item in exceptions %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: {{ _('Sub-Modules') }}
.. autosummary::
   :toctree:  {{ name }}
   :template: custom-module-template.rst
   :recursive:
    {% for item in modules %}
       {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
