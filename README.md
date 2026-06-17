# Jaxon

Jaxon is a Python library for saving and loading arbitrary nested data structures
("pytrees") to the [Hierarchical Data Format (HDF5)](https://wikipedia.org/wiki/Hierarchical_Data_Format).

HDF5 is an open, self-describing format with native support for multidimensional
arrays and metadata. Jaxon stores enough information to reconstruct the original
Python objects, so HDF5 files produced by Jaxon can be inspected with standard
tools such as `h5dump` or `HDFView`, and can be read even when the original code
is no longer available.

Jaxon is well suited for machine learning and scientific computing. It is especially
compatible with packages that use Python dataclasses and
[JAX](https://github.com/jax-ml/jax), such as [Equinox](https://docs.kidger.site/equinox/).

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

`save` also accepts a file-like object instead of a path:

```python
import tempfile
with tempfile.TemporaryFile() as f:
    save(f, pytree)
    result = load(f)
```


## Supported types

A pytree is any nested combination of the types listed below. Dictionary keys may
themselves be pytrees (as long as they are hashable). Circular references are
detected and raise an error.

| Type | Stored as |
| ---- | --------- |
| `list`, `tuple`, `dict`, `set`, `frozenset` | HDF5 group |
| `str` | HDF5 UTF-8 fixed-length string |
| `int`, `float`, `bool`, `complex` | String representation (see [Python numerics](#python-numeric-types)) |
| `None`, `Ellipsis`, `slice`, `range` | String representation |
| `bytes`, `bytearray`, `memoryview` | HDF5 attribute (or dataset) |
| `np.ndarray`, `jax.Array` | HDF5 attribute (or dataset) |
| `np.bool_` | HDF5 attribute |
| `np.int8`, `np.int16`, `np.int32`, `np.int64` | HDF5 attribute |
| `np.uint8`, `np.uint16`, `np.uint32`, `np.uint64` | HDF5 attribute |
| `np.float16`, `np.float32`, `np.float64` | HDF5 attribute |
| `np.complex64`, `np.complex128` | HDF5 attribute |
| `np.longdouble`, `np.clongdouble` | HDF5 attribute (see [Platform-specific types](#platform-specific-numeric-types)) |
| Any Python dataclass | HDF5 group containing all fields (see [Dataclasses](#dataclasses)) |

Only the array contents are stored; metadata such as array titles is not preserved.


### Python numeric types

By default Jaxon preserves the Python types `int`, `float`, `bool`, and `complex`
exactly using a string representation. To store them as compact binary HDF5
attributes (which is more efficient for large numbers of scalars) pass
`exact_python_numeric_types=False` to `save`:

```python
save("data.hdf5", {"x": 1, "y": 3.14}, exact_python_numeric_types=False)
result = load("data.hdf5")
# result["x"] is np.int64(1), result["y"] is np.float64(3.14)
```

To convert only specific Python types, use `py_to_np_types`:

```python
save("data.hdf5", data, py_to_np_types=(int, float))  # bool and complex stay as strings
```


### Platform-specific numeric types

`np.longdouble` and `np.clongdouble` are supported on all platforms, but their
precision depends on the hardware:

- **Linux x86-64**: 80-bit extended precision (stored in 128 bits); accessible
  also as `np.float128` and `np.complex256`.
- **Windows / macOS ARM**: 64-bit (same precision as `np.float64`); `np.float128`
  does not exist on these platforms.

A file containing `np.longdouble` scalars saved on Linux can be loaded on Windows,
but values will be truncated to 64-bit precision. Jaxon does not warn about this
because the truncation happens inside h5py.


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

For types you cannot modify, pass callables to `save` and `load`:

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

Multiple marshalers can be provided; the first one returning non-`None` is used.
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

By default all arrays, byte buffers, and memoryviews are stored as HDF5
attributes. For very large arrays it can be preferable to use HDF5 datasets
instead, which support chunking and compression:

```python
from jaxon import save, JaxonStorageHints

big_array = np.zeros((1000, 1000))
save(
    "data.hdf5",
    {"array": big_array},
    storage_hints=[(big_array, JaxonStorageHints(store_in_dataset=True))],
)
```

The hint is identified by object identity (`is`), so the object passed in
`storage_hints` must be the same object that appears in the pytree.


## Acknowledgements

Jaxon is built on the following libraries:

- [NumPy](https://numpy.org/)
- [JAX](https://github.com/jax-ml/jax)
- [h5py](https://www.h5py.org/)
- [dill](https://dill.readthedocs.io/en/latest/)
