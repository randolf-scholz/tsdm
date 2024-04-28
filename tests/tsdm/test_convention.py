"""Test naming conventions for TSDM.

Conventions:

1. Naming Schema:

    - Protocols: <Kind> or <Kind>Protocol
    - Abstract base classes: <Kind>ABC or <Kind>Base
    - Structural classes: <Kind><Name>
    - Particular classes: <Name><Kind> or <Name>
    - Functions: <kind>_<name> or <name>

2. Type conventions:

    - functions of schema `is_<cond>` return a boolean or `TypeGuard`.
"""
