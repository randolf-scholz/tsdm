{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
    :members:

    {% block modules %}
    {% if modules %}
    .. rubric:: {{ _('Sub-Modules') }}
    .. autosummary::
       :toctree: {{ fullname }}
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
      :toctree: {{ fullname }}
    {% for item in attributes %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block functions %}
    {% if functions %}
    .. rubric:: {{ _('Module Functions') }}
    .. autosummary::
      :toctree:
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
      :toctree: {{ fullname }}
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
      :toctree: {{ fullname }}
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
      :toctree: {{ fullname }}
    {% for item in members %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
    #}

