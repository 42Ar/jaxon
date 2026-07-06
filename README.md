# Jaxon

[![Tests](https://github.com/42Ar/jaxon/actions/workflows/tests.yml/badge.svg)](https://github.com/42Ar/jaxon/actions/workflows/tests.yml)

Jaxon is a focused Python library for saving and loading arbitrary nested data structures
("pytrees") to the [Hierarchical Data Format (HDF5)](https://wikipedia.org/wiki/Hierarchical_Data_Format).

HDF5 is an open, self-describing format with native support for multidimensional
arrays and metadata. Jaxon stores enough information to reconstruct the original
Python objects, so HDF5 files produced by Jaxon can be inspected with standard
tools such as `h5dump` or `HDFView`, and can be read even when the original code
is no longer available.

Jaxon is well suited for machine learning and scientific computing. It is especially
compatible with packages that use Python dataclasses and
[JAX](https://github.com/jax-ml/jax), such as [Equinox](https://docs.kidger.site/equinox/).
Jaxon intentionally has a narrow scope — saving and loading pytrees is all it does.
The `save` and `load` API is stable.

**Requires Python ≥ 3.12.**


## Installation

```bash
pip install jaxon
```


## Quick start

```python
from jaxon import save, load
import numpy as np
import jax.numpy as jnp

pytree = {
    "mylist": ["foo", "bar", 42],
    "myset": {"a", "b", "z", (42, b"blob")},
    "numpy_array": np.arange(3),
    "jax_array": jnp.arange(3),
}
save("data.hdf5", pytree)
print(load("data.hdf5"))
```
```
{'mylist': ['foo', 'bar', 42], 'myset': {'b', 'a', (42, b'blob'), 'z'}, 'numpy_array': array([0, 1, 2]), 'jax_array': Array([0, 1, 2], dtype=int32)}
```


## Supported types

Jaxon can save and load all pytrees that are nested structures of the types listed below. Note that custom types can be added, see [Custom types](#custom-types).
Dictionary keys may themselves be pytrees (as long as they are hashable).
Circular references are detected and raise an error.

| Type                                        | Stored as                                                            |
|---------------------------------------------|----------------------------------------------------------------------|
| `list`, `tuple`, `dict`, `set`, `frozenset` | HDF5 group                                                           |
| `str`                                       | HDF5 UTF-8 fixed-length string                                       |
| `int`, `float`, `bool`, `complex`           | String representation (see [Python numerics](#python-numeric-types)) |
| `None`, `Ellipsis`, `range`                 | String representation                                                |
| `bytes`, `bytearray`                        | HDF5 dataset                                                         |
| All numpy generics ¹                        | HDF5 attribute                                                       |
| `np.ndarray`, `jax.Array` ²                 | HDF5 dataset                                                         |
| Python dataclasses                          | HDF5 group                                                           |

¹ Except for `numpy.object_`

² Only the actual data of the array is stored. Metadata such as column titles is not preserved. Arrays can have dataypes of the supported numpy generics.


### Dataclasses

Jaxon stores the module name, class name, and all field values of a dataclass.
On load, the class is instantiated via `__new__` (bypassing `__init__`) and each
field is set directly, which works even for frozen dataclasses.

```python
from dataclasses import dataclass
import numpy as np
from jaxon import save, load

@dataclass
class Model:
    weights: np.ndarray
    bias: float
    name: str

m = Model(weights=np.array([1.0, 2.0]), bias=0.5, name="linear")
save("model.hdf5", m)
print(load("model.hdf5"))  # Model(weights=array([1., 2.]), bias=0.5, name='linear')
```

Machine learning packages such as [Equinox](https://docs.kidger.site/equinox/)
automatically make all modules Python dataclasses, so Jaxon is fully compatible
with them.


#### Schema evolution

If a dataclass has changed between saving and loading (fields added or removed),
the following options control behaviour:

```python
result = load(
    "model.hdf5",
    allow_missing_fields=True,   # fields in file but absent from the class: warn, skip
    allow_unknown_fields=True,   # fields in class but absent from file: use default or JAXON_NOT_LOADED
)
```


### Python numeric types

By default Jaxon preserves the Python types `int`, `float`, `bool`, and `complex`
exactly. If `exact_python_numeric_types=False` is passed to `save`, they are converted to numpy generics before saving and stored in binary form.
This also implies that they are loaded as numpy generic:

```python
save("data.hdf5", {"x": 1, "y": 3.14}, exact_python_numeric_types=False)
result = load("data.hdf5")
# result["x"] is np.int64(1), result["y"] is np.float64(3.14)
```

To convert only specific Python types, use `py_to_np_types`:

```python
save("data.hdf5", data, py_to_np_types=(int, float))  # bool and complex stay as strings
```


## Reference identity

Jaxon preserves reference identity across a save/load cycle. If the same object
appears at multiple locations in a pytree, it will be the same object after loading:

```python
a = np.array([1, 2, 3])
pytree = {"x": a, "y": a}
save("data.hdf5", pytree)
result = load("data.hdf5")
assert result["x"] is result["y"]  # True
```


## Custom types

### `to_jaxon` / `from_jaxon` interface

For types that are not natively supported and not dataclasses, implement
`to_jaxon` and `from_jaxon`:

```python
class MyModel:
    def __init__(self, weights, config):
        self.weights = weights
        self.config = config

    def to_jaxon(self):
        # return a supported container
        return {"weights": self.weights, "config": self.config}

    def from_jaxon(self, data):
        # populate self from the container returned by to_jaxon
        self.weights = data["weights"]
        self.config = data["config"]

save("model.hdf5", MyModel(np.eye(3), {"lr": 0.01}))
result = load("model.hdf5")  # MyModel instance
```

`to_jaxon` takes priority over the dataclass fallback if both apply. Jaxon stores
the fully-qualified class name so the correct class is instantiated on load.


### Custom marshaler/unmarshaler functions

For types that cannot be modified, callables can be passed to `save` and `load`:

```python
def marshal(obj):
    if isinstance(obj, MyClass):
        return "mymodule.MyClass", {"value": obj.value}
    return None  # signal that this marshaler cannot handle the type

def unmarshal(qualified_name, data):
    if qualified_name == "mymodule.MyClass":
        return MyClass(data["value"])
    return None

save("data.hdf5", obj, custom_marshalers=[marshal])
result = load("data.hdf5", custom_unmarshalers=[unmarshal])
```

If multiple marshalers are provided the first that returns a non-`None` result is used.
Custom marshalers take priority over `to_jaxon`/`from_jaxon` and the dataclass
fallback.


### Serialization with dill

As a last resort, Jaxon can serialize unsupported types using
[dill](https://dill.readthedocs.io/en/latest/) (an extended pickle) and store the
result as a binary blob. This must be opted into explicitly:

```python
save("data.hdf5", obj, allow_dill=True)
result = load("data.hdf5", allow_dill=True)
```

Note that dill-serialized objects are opaque binary blobs and cannot be inspected
with HDF5 viewers.


## Partial loading with `load_filter`

Large pytrees can be partially loaded by providing a filter function. The filter
receives the path to each node as a list and returns `True` to load it or `False`
to skip it (skipped nodes are replaced with `JAXON_NOT_LOADED`):

```python
from jaxon import load, has_common_prefix, JAXON_NOT_LOADED

# only load pytree["weights"] and anything nested under it
result = load("model.hdf5", load_filter=lambda path: has_common_prefix(["weights"], path))

result["weights"]  # np.ndarray — loaded
result["config"]   # JAXON_NOT_LOADED — skipped
```

`has_common_prefix(prefix, path)` is a convenience function that returns `True`
when `path` starts with `prefix`. For dictionaries the path element is the dict key,
for lists and sets it is the integer index, and for dataclasses it is the field name
string. Dict keys themselves are always loaded regardless of the filter.


## Storage hints


## Schema

This section describes how Jaxon stores pytrees in an HDF5 file.


### Overall Design Rational

1. Jaxon adheres to the principle of storing data efficiently wherever the user provides the data in an efficiently packed format (e.g. as an numpy array).
In other cases (e.g. for single int attributes, lists of Dataclasses, etc.) Jaxon prefers portability and simplicity over efficiency.
2. Every stored data item (e.g. dictionary entry, list entry, dataclass attribute, etc.) is represented by a single attribute.


### Python containers

####  dict


#### list, tuple, set frozenset


### Scalar NumPy Types

Jaxon supports the following numeric NumPy generic types:

| numpy dtype                | ctype               | Native HDF5 Type           |
|----------------------------|---------------------|----------------------------|
| numpy.bool_                | bool                | H5T_NATIVE_HBOOL           |
| numpy.byte                 | char                | H5T_NATIVE_SCHAR           |
| numpy.ubyte                | unsigned char       | H5T_NATIVE_UCHAR           |
| numpy.short                | short               | H5T_NATIVE_SHORT           |
| numpy.ushort               | unsigned short      | H5T_NATIVE_USHORT          |
| numpy.intc                 | int                 | H5T_NATIVE_INT             |
| numpy.uintc                | unsigned int        | H5T_NATIVE_UINT            |
| numpy.uint / numpy.ulong   | ¹                   | ¹                          |
| numpy.int_ / numpy.long    | long                | H5T_NATIVE_LONG            |
| numpy.longlong             | long long           | H5T_NATIVE_LLONG           |
| numpy.ulonglong            | unsigned long long  | H5T_NATIVE_ULLONG          |
| numpy.half / numpy.float16 | _Float16            | H5T_NATIVE_FLOAT16         |
| numpy.single               | float               | H5T_NATIVE_FLOAT           |
| numpy.double               | double              | H5T_NATIVE_DOUBLE          |
| numpy.longdouble           | long double         | H5T_NATIVE_LDOUBLE         |
| numpy.csingle              | float complex       | H5T_NATIVE_FLOAT_COMPLEX   |
| numpy.cdouble              | double complex      | H5T_NATIVE_DOUBLE_COMPLEX  |
| numpy.clongdouble          | long double complex | H5T_NATIVE_LDOUBLE_COMPLEX |

¹ According to the numpy 2.5 documentation, the datatype is 64bit on 64bit systems and 32 bit on 32 bit systems and therefore has no definitive c equivalent.

Note that the actual width of each type depends on the specifc platform: An array saved as `numpy.long` on one platform might load as an `numpy.intc` on another platform.
Only the width and signedness of the type is guranteed to be equal on both platforms.
The types listed above are stored directly in the attribute value without appending a typehint to the respective key.
All other types either store a typehint appended to the respective attribute key; or they are guranteed to be of string type.

Additionally, the following strings types are supported:

|numpy dtype  | HDF5 Type  | HDF5 Character Set | Jaxon typehint |
|-------------|------------|--------------------|----------------|
|numpy.bytes_ | H5T_STRING | H5T_CSET_ASCII     | numpy.bytes_   |
|numpy.str_   | H5T_STRING | H5T_CSET_UTF8      | numpy.str_     |


### Numpy Arrays


### Jax Arrays

Jaxon supports n-dimensional numpy arrays of the types above. Jax arrays are first converted to numpy. The type information is preserved by adding the typehint `jax.Array`.


### Python types

The following python types are stored by converting them to utf8 strings (H5T_STRING type with H5T_CSET_UTF8). The representation is mostly the same as produced by python's `repr(...)` function.

| python type | Encoding                                                                                  |
|-------------|-------------------------------------------------------------------------------------------|
| bool        | `repr(...)` of the bool                                                                   |
| int         | `repr(...)` of the int                                                                    |
| float       | `repr(...)` of the float                                                                  |
| complex     | `repr(...)` of the complex; brackets are always present even for purely imaginary numbers |
| str         | raw string, quouted with single qoutes                                                    |
| None        | `None`                                                                                    |
| Ellipsis    | `Ellipsis`                                                                                |
| range       | `repr(...)` of the range                                                                  |

The rational behind encoding these python datatypes as strings is that python dictionaries map most naturally to HDF5 Groups, where the keys must be provided as strings.
Since dictionary keys are often of type `int` or `str`, encoding them as strings results in the most convenient format. For consistency, it was decided to use the same encoding also for cases where the value is not a dicitonary key.

Furthermore, the following python types are supported by converting them to one of the numpy type above and adding a typehint to preserve the original type information.  

|python type   | converted to                | typehint   |
|--------------|-----------------------------|------------|
|`bytes`       | numpy.bytes_                | bytes      |
|`bytearray`   | numpy.ndarray (dtype=uint8) | bytearray  |


### Root Node


### Datclasses


### Chained type conversion

## Acknowledgements

Jaxon is built on the following libraries:

- [NumPy](https://numpy.org/)
- [JAX](https://github.com/jax-ml/jax)
- [h5py](https://www.h5py.org/)
- [dill](https://dill.readthedocs.io/en/latest/)
