# TODOs

##

- Encoder serialization / deserialization (beyond pickle)
  - use a library like pydantic/attrs to convert to JSON
  - What about the "params" tuple
- Multiple parametrizations
- samplers / dataloading
- tasks (MIMIC-III, MIMIC-IV)
- custom tasks with metadata.
- encoders: optimal transport, vector encoders
- factorized filter (controls treated differently)

For encoders and samplers, we want some dataclass like library.
Requirements:

- annotated attributes should be converted to `__slots__` of the class
- We often want to use custom `__init__`, as we allow wider input types than the annotated type, i.e. we usually want to perform some kind of type casting.
- In particular, we want to allow string input such as "3h", instead of requiring a `timedelta` object for convenience.
- Initialized classes might not be final, in particular encoders require a `fit` method.
  Only after calling `fit` the class should be considered final.
- Classes should have a fixed set of attributes, i.e. we want to use dataclasses with `__slots__`.
- Finalized objects should be serializable to a stable text format, e.g. JSON/YAML/YOML.
  - Serialization format should natively support datetime and timedelta objects.
  - This should work recursively, as an encoder might use another encoder as a submodule
- Serialized object should be deserializeable, i.e. we want to be able to load a serialized object and call `fit` on it.
- The base class/decorator should avoid adding extra attributes.
- Support for type hints.
- when converting to `dict`, special fields should be stored:
  - `__name__`: name of the class
  - `__module__`: module of the class
    This allows us to reconstruct the class from the dictionary:
  - `getattr(module, name)(**dict)`

## Converting class to dictionary

This is needed for initializing classes from hyperparamters.
Essentially, each module should have a hyperparameter dictionary with defaults.
Some hyperparameters might be dependent

When converting a class to a dictionary, then fields without default value are marked with `NotImplemented`.
Note that this

- finally, we want to write hyperparameter dictionaries for torch modules.
  In this case, we want a dictionary-like object for each class,
  that can be used to initialize the class.
  If a class requires submodules, then `__init__` should expect these modules as arguments.
  an additional `from_hyperparams` classmethod should be provided, which should accept a single (typed) dictionary as input.
  This dicitonary should be used to initialize the class and its submodules.

So:

1. Each finalized object is associated with a dataclass/typeddict like object.
2. This object uses strict data types, whereas `__init__` might be more flexible.
3. Objects should be initializable given a (nested) dictionary that matches the dataclass/typeddict.

There are 3 types of fields:

1. Mandatory fields
2. Optional fields. These may or may bot be present (could be `None`/`null`)
3. Dependent fields. These are computed from other fields. They are not mandatory, but also do not have a fixed default value.

|                 | input to `__init__`? | has a static default? |
| --------------- | -------------------- | --------------------- |
| default field   | mandatory            | no                    |
| optional field  | yes                  | yes                   |
| dependent field | yes                  | no                    |
| computed field  | no                   | no                    |

- Optional fields should be annotated as `field_name: Optional[field_type] = None`
- Dependent fields should be annotated as

Example: Encoder:

- needs some inputs -> `__init__`
- after calling `encoder.fit(data)` it is finalized
  - we should be able to export/import non-finalized encoders as well as finalized encoders
  - There are attributes which are only set after calling `fit`, e.g. `encoder._input_dim`
  - When deserializing, all these attributes need to be loaded correctly.
    - However, we may not wish to show all of these attributes in documentation, or have them as individual fields.
    - Possibly, these attributes could be collected in a dictionary "auxiliary fields" which is not part of the dataclass.

---

### OLD TODOs

- Rezero as parameterization
- Rezero class
- Generic classes
- StandardScalar is not properly vectorized. Do custom implementation instead.
- Exact loss function? Forecast?
- Without Standard-scaling
- RMSE **NOT** channel wise, but flattened
- accumulate cross-validation: RMSE
- horizon 1h / 2h / 6h ↭ avg number of timestamps
- horizons: 150+100, 300+300, 400+600
- In TS split: first snippet that fits until last snippet
- Time Encoding?
- negative values: truncate to zero!

```
>>> timeseries.min()
Flow_Air                           0.000000
StirringSpeed                      0.000000
Temperature                       32.689999
Acetate                           -0.257518
Base                               0.000000
Cumulated_feed_volume_glucose      3.000000
Cumulated_feed_volume_medium       5.882958
DOT                                0.000000
Glucose                           -0.094553
OD600                             -0.962500
Probe_Volume                     200.000000
pH                                 0.000000
Fluo_GFP                        -250.000000
InducerConcentration               0.000000
Volume                            -2.235569
```

Current WorkFLow

KIWI data - cut off negative values

- push tsdm with KIWI task
- run linodenet on KIWI data
- create graphs
- encode metadata
- results with metadata
- results on Electricity, ETTh, Traffic

ADD DATASETS:

- ECL: <https://drive.google.com/file/d/1rUPdR7R2iWFW-LMoDdHoO2g4KgnkpFzP/view?usp=sharing>
- Weather: <https://drive.google.com/file/d/1UBRz-aM_57i_KCC-iaSWoKDPTGGv6EaG/view?usp=sharing>
