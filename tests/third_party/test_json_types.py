# type JSON_SCALAR = str | bool | int | float | None
# type JSON = dict[str, JSON_SCALAR | list[JSON] | dict[str, JSON]]


# type JSON_SCLAR = str | int | float | bool | None
# type JSON_TYPE = JSON_SCLAR | dict[str, JSON_TYPE] | list[JSON_TYPE]
# type JSON = dict[str, JSON_TYPE]

type JSON_SCALAR = None | bool | int | float | str
type JSON_ARRAY[T: JSON = JSON] = list[T]
type JSON_DICT[T: JSON = JSON] = dict[str, T]
type JSON = JSON_SCALAR | JSON_ARRAY[JSON] | JSON_DICT

# no inductive types in python ...
type JSON_TENSOR = (
    list[float]
    | list[list[float]]
    | list[list[list[float]]]
    | list[list[list[list[float]]]]
    | list[list[list[list[list[float]]]]]
    | list[list[list[list[list[list[float]]]]]]
    | list[list[list[list[list[list[list[float]]]]]]]
    | list[list[list[list[list[list[list[list[float]]]]]]]]
)

type CFG_SCALAR = None | bool | int | float | str
type CFG_ARRAY[T: CFG_SCALAR] = list[T]


# theoretically, we would need something like this:
# type Foo = float | Map[list, Foo]
# because then
# type Foo = float | Map[list, float | Map[list, Foo]]
#          = float | Map[list, float] | Map[list, Map[list, Foo]]]
#          = float | list[float] |

d: JSON = {
    "input_size": 64,
    "output_size": 64,
    "bias": True,
    "activation": {
        "__name__": "ReLU",
        "__module__": "torch.nn",
        "inplace": False,
    },
    "shape": [64, 64],
}
