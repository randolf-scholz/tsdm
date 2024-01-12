#!/usr/bin/env python


class A:
    a = 1

    def __getattribute__(self, item):
        return object.__getattribute__(self, item)


A().a
