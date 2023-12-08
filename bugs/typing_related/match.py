#!/usr/bin/env python

x: str | tuple[str, str] = "a"

match x:
    case "a" | ("a", "a"):
        print("common case")
    case ("a", "b"):
        print("mixed case")
    case _: ...
