#!/usr/bin/env python
# https://github.com/PyCQA/pylint/issues/8082


assert isinstance(0, int | str)
