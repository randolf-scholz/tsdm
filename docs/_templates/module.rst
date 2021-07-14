{{ fullname | escape | underline}}


.. automodule:: {{ fullname }}

    {% block modules %}
    {% if modules %}
    .. rubric:: Modules

    .. autosummary::
        :toctree: {{ fullname }}
        :template: module.rst
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
    {% if modules %}


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
        :template: class.rst
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
    {% else %}
    {% endif %}
