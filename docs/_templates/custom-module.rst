{{ fullname | escape | underline}}


.. automodule:: {{ fullname }}
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

    {% block modules %}
    {% if modules %}
    .. rubric:: Modules

    .. autosummary::
        :toctree: {{ fullname }}
        :template: custom-module.rst
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
    .. rubric:: {{ _('Functions') }}

    .. autosummary::
        :toctree: {{ fullname }}
    {% for item in functions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block classes %}
    {% if classes %}
    .. rubric:: {{ _('Classes') }}

    .. autosummary::
        :toctree: {{ fullname }}
        :template: custom-class.rst
    {% for item in classes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block exceptions %}
    {% if exceptions %}
    .. rubric:: {{ _('Exceptions') }}

    .. autosummary::
        :toctree: {{ fullname }}
    {% for item in exceptions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

