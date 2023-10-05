# Technical Note

## Naming Conventions

- `is_<cat>(x)`-methods:
  - check if the object has a certain property or belongs to a certain category
  - must return a boolean
  - prefer `is_<cat-with-property>(x)` over `has_<property>(x)`-methods
- `*_to_*`-methods: convert from one representation to another
- `*_from_*`-methods: convert from one representation to another
- `as_<type>(x)`-methods: cast object to type.
- `self.get_*`-methods: return a value that's cheap to compute or already exists
- `self.set_*`-methods: set a value that's cheap to compute or already exists
- `self.make_*`-methods: return a new object that's expensive to compute / doesn't exist yet
- `cls.from_*`-methods: classmethods that return a new object from data
  - must return a `Self`.
