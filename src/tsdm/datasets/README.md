# Some Notes for Schema

## Which file format

Requirements:

- store multiple table like data structures
- need support for basic types (int, float, string, bool, null)
- need support for lists, tuples and dicts
- need support for comments
- need support for datetime and timedelta types

Options:

- json
- yaml
- toml
- protobuf
- python

We decided to use YAML for the following reasons:

- better readable than json
- supports mixed type lists/tuples (toml does not)
- supports null (toml does not)

However, yaml is not perfect:

- can mush types (always use single quotes for strings!!!)
