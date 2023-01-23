#!/usr/bin/env python
# https://github.com/PyCQA/pylint/issues/8082


class Foo:
    default_path = "..."  # not to be specified in __init__

    def __init__(self, file):
        self.path = self.default_path + file

    @classmethod
    def from_alt_path(cls, alternative_path, *args, **kwargs):
        obj = super().__new__(cls)
        obj.default_path = alternative_path
        obj.__init__(*args, **kwargs)  # ✘ unnecessary-dunder-call
        return obj
