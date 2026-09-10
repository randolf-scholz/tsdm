# About numeric types package

This subpackage is a - partially failed - attemtpt to write some generic `Protocol`
classes for numerical data types in Python such as `np.ndarray`, `torch.Tensor`, `pd.Series`, etc.

At the time of writing, several limitations in the type system make it virtually impossible
to satisfy the goals of this subpackage, which are/were to create generic protocols for:

- Scalars (0-dimensional and convertible to python scalars (int, float, complex, bool))
- Arrays/Tensors (N-dimensional with homogeneous data type)
- Series: 1-dimensional with homogeneous data type; iteration yields scalars
- Tables/DataFrames: 2-dimensional with heterogeneous column types.

such that type checkers could automatically determine the correct types when using
classes that implement these protocols:

```python
def positive_values(arg: FloatArray) -> BoolArray:
    return arg > 0


positive_values(np.array([1.0, -2.0, 3.0]))  # np.ndarray of dtype bool
```

However, there are several limitations that make this difficult or impossible:

- <https://github.com/python/typing/issues/2021> makes implementing Array for
  Timedelta/Datetime impossible quasi impossible, because different libraries may
  use different overload order for `__add__` and `__sub__`, e.g.

  ```python
  from typing import overload


  class Duration: ...


  class Timestamp:
      @overload
      def __sub__(self, other: Duration) -> Timestamp: ...
      @overload
      def __sub__(self, other: Timestamp) -> Duration: ...
   ```

  vs.

  ```python
  from typing import overload


  class Duration: ...


  class Timestamp:
      @overload
      def __sub__(self, other: Timestamp) -> Duration: ...
      @overload
      def __sub__(self, other: Duration) -> Timestamp: ...
  ```

  Moreover, note that if the classes are not `@disjoint_bases`, then these overloads
  are not even guranteed to be exchangable, since a type checker has then to assume
  a mutual subclass is possible.

- To make the `positive_values` example above work as intended, we would need
  higher-kinded types (HKTs) / be able to upper bound type variables with generic types, e.g.

  ```python
  # protocol needs to know the "partner" type (ndarray[bool] for ndarray[float],
  # Tensor[bool] for Tensor[float], etc.) for boolean operations.
  class FloatArray[MaskType: Boolarray]: ...


  def positive_values[Ret: BoolArray, Arg: FloatArray[Ret]](arg: Arg) -> Ret:
      return arg > 0
  ```

- From a convenience perspective, writing such generic code that can autodetect the used
  library (numpy, torch, pandas, etc.) questionable at best, because we cannot parametrize
  individual paramters of a generic type. For instance in the example above, a real world
  `FloatArray` protocol would likely need to be parametrized with many partner types,
  but writing out all partner types like `FloatArray[BoolArray, IntArray, ComplexArray, ...]`
  would be cumbersome and make formulas virutually unreadable. This would only work
  a little bit if we could select the necessary partner types by key lookup on a per
  function basis

  ```python
  class FloatArray[MaskType: Boolarray]: ...

  def positive_values[Ret: BoolArray, Arg: FloatArray[MaskType=Ret]](arg: Arg) -> Ret:
      return arg > 0
  ```
