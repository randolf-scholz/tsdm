{{ objname | escape | underline}}

.. automodule:: {{ fullname }}
    :members:

    {% block modules %}
    {% if modules %}
    .. rubric:: {{ _('Sub-Modules') }}
    .. autosummary::
       :toctree: {{ objname }}
       :template: custom-module-template.rst
       :recursive:
    {% for item in modules %}
       {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Module Attributes') }}
    .. autosummary::
      :toctree: {{ objname }}
    {% for item in attributes %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}
    .. rubric:: {{ _('Module Functions') }}
    .. autosummary::
      :toctree: {{ objname }}
      :nosignatures:
    {% for item in functions %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block classes %}
    {% if classes %}
    .. rubric:: {{ _('Module Classes') }}
    .. autosummary::
      :toctree: {{ objname }}
      :template: custom-class-template.rst
      :nosignatures:
    {% for item in classes %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block exceptions %}
    {% if exceptions %}
    .. rubric:: {{ _('Module Exceptions') }}
    .. autosummary::
      :toctree: {{ objname }}
    {% for item in exceptions %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {# Do not use this block for now
    {% block members %}
    {% if members %}
    .. rubric:: {{ _('Module Members') }}
    .. autosummary::
      :toctree: {{ objname }}
    {% for item in members %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
    #}

