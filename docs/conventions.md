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

## Computed fields in dataclasses

Standard python dataclasses do not know the notion of a "computed" field.
Quite often, we want that either:

  1. the user can provide the value of a field,
  2. if the don't, the field is computed from other fields.

Moreover, computed fields interplay with both typing and serialization.
Computed fields are usually essential in the actual usage of the object.
We distinguish between 2 types of derived fields:

  1. Fields that can be determined at initialization time.
  2. Fields that require additional data to be computed. This applies in particular to models such as sklearn transforms.

For this purpose, we introduce type qualifiers that more precisely let us define the behavior of the fields.
Moreover, we use metaclasses to inspect these fields, and to automatically validate the object.

Typically, we want these fields to be optional arguments in the constructor.
Therefore, they need to be given a default value.

However, for typing purposes, it is annoying having to deal with values being `None`, when we finally use the fully initialized object.
To alleviate these issues, we use a sentinel values `MISSING` / `UNDEFINED` / `NOT_SET` / `NOT_GIVEN` to indicate that the fields are not set.

Regarding the default value, there is one potential problem:

- if we want to allow serializing it, then we need to provide a default value.
- the only sensible value is `None`, as it is the only value that can be serialized to JSON.
- however, `None` might be a valid value for the field, which would lead to ambiguity.
- this case is only consistent if `None` is also the default value for the field.
- this in turn mean we would have to perform type analysis to determine whether the model is serializable
- also how do we accomodate users that do not use type hints?
- therefore, we may want to use a value different than `None`, but substitute it with `None` when serializing.

### Serialization

Models should still be serializable as long as all non-optional fields are given. For the optional fields, there are two options:

1. serialize the field, providing some default value.
2. do not serialize the field.

In some cases, it may be serialable to make models JSON-serializable, which only supports a very small amount of types.
