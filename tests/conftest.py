r"""Configuration for pytest."""

import pytest


def pytest_collection_modifyitems(config, items):  # ruff: ignore[ARG001]
    interactive = []
    non_interactive = []
    for item in items:
        if any(item.get_closest_marker(mark) for mark in INTERACTIVE_MARKS):
            interactive.append(item)
        else:
            non_interactive.append(item)

    # If the current selection contains at least one non-interactive test,
    # skip interactive ones.
    #
    # This means:
    # - "pytest"            -> interactive tests skipped
    # - "pytest tests/x.py" -> if mixed, interactive skipped
    # - "pytest path::test_my_plot" -> only interactive selected, so it runs
    # - "pytest -k plot"    -> if selection resolves only to interactive tests, they run
    interactive_test_ids = {
        (item.parent.nodeid, getattr(item, "originalname", None) or item.name)
        for item in interactive
    }
    if non_interactive or len(interactive_test_ids) > 1:
        for item in interactive:
            reason = (
                "manual tests must be run individually"
                if item.get_closest_marker(MANUAL_MARK)
                else "interactive tests must be run individually"
            )
            skip = pytest.mark.skip(reason=reason)
            item.add_marker(skip)


INTERACTIVE_MARK = "interactive"
MANUAL_MARK = "manual"
INTERACTIVE_MARKS = (INTERACTIVE_MARK, MANUAL_MARK)
