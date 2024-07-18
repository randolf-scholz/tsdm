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

## Timeseries Descriptions

For a given table, we want to describe the variables, i.e. columns of the table.
We consider several attributes of interest:

| name        | dtype      | description                                     | examples                                                     |
|-------------|------------|-------------------------------------------------|--------------------------------------------------------------|
| dtype       | `cat[str]` | The data-type associated with the variable      | `int32`, `float64`, `cat[str]`                               |
| unit        | `cat[str]` | The physical unit associated with the varaible. | `kg`, `%`, `‰`                                               |
| type        | `cat[str]` | Arbitrary category to assign to the variable    | `control`, `observable`                                      |
| scale       | `cat[str]` | The scale the variable lives on.                | `absolute`, `linear`, `logarithmic`, `percentage`            |
| bounds      | `interval` | The range of valid values of the variable.      | `[0,14]` (pH), `[0,∞)`: weight, `[−273.15,∞)` (temp in `°C`) |
| description | `str`      | Description of the variable.                    |                                                              |

- `dtype (cat[str])`: The data-type associated with the variable
- `unit (cat[str])`: The unit of the variable. **Note:** dimensionless quantities should be represented by the unit `1`.
- `type (cat[str])`: The type of the variable (arbitrary classification).
- `description (str)`: A description of the variable
- `scale (cat[str])`: The scale of the variable (applicable to real-valued variables)

## Style Guide

When to use `@property`:

- Accessing the property should not have user-facing side-effects.
- Accessing the property multiple times directly in a row should return the same value.
  (i.e. `A.foo == A.foo` must always be true, however `x = A.foo; A.bar(); x == A.foo` may be false)
- If it is computationally expensive to compute the property, caching should be used.
  (i.e. Calling A.foo N times should cost less than N times the cost of computing A.foo once)
