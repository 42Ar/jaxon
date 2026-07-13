# Jaxon

[![Tests](https://github.com/42Ar/jaxon/actions/workflows/tests.yml/badge.svg)](https://github.com/42Ar/jaxon/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/42Ar/jaxon/branch/main/graph/badge.svg)](https://codecov.io/gh/42Ar/jaxon)

Jaxon is a focused Python library for saving and loading arbitrary nested data structures
("pytrees") to the [Hierarchical Data Format (HDF5)](https://wikipedia.org/wiki/Hierarchical_Data_Format).

HDF5 is an open, self-describing format with native support for multidimensional arrays
and metadata. Jaxon stores enough information to reconstruct the original Python objects,
so HDF5 files produced by Jaxon can be inspected with standard tools such as `h5dump` or
`HDFView`, and can be read even when the original code is no longer available.

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
save("pytree.hdf5", pytree)
print(load("pytree.hdf5"))
```
```
{'mylist': ['foo', 'bar', 42], 'myset': {'b', 'a', (42, b'blob'), 'z'}, 'numpy_array': array([0, 1, 2]), 'jax_array': Array([0, 1, 2], dtype=int32)}
```


## Supported types

Jaxon can save and load all pytrees that are nested structures of the types listed
below, as well as [custom types](#custom-types). Note that dictionary
keys may themselves be pytrees. Circular references are detected and raise an error.

|Type                                        |Stored as                                                            |
|--------------------------------------------|---------------------------------------------------------------------|
|`list`, `tuple`, `dict`, `set`, `frozenset` |HDF5 group                                                           |
|`str`                                       |HDF5 utf-8 fixed-length string                                       |
|`int`, `float`, `bool`, `complex`           |String representation (see [Python numerics](#python-numeric-types)) |
|`None`, `Ellipsis`, `range`                 |String representation                                                |
|`bytes`, `bytearray`                        |HDF5 opaque type                                                     |
|All NumPy scalars, including `bool_`        |Corresponding native HDF5 type                                       |
|`np.str_`, `np.bytes_`, `np.void`           |Corresponding native HDF5 type                                       |
|`np.ndarray` ¹, `jax.Array`                 |Compatible native HDF5 type                                          |
|Python dataclasses                          |HDF5 group                                                           |

¹ The actual data and the datatype including field names are stored. Column titles are not preserved. Fields of type `np.object_` and `np.str_` are not supported.


### Dataclasses

Jaxon stores the module name, class name, and all field values and names. On load, the class
is instantiated via `__new__` (bypassing `__init__`). Then, each field is set in a way that is
compatible with frozen dataclasses.

```python
from dataclasses import dataclass
import jax
from jaxon import save, load

@dataclass(frozen=True)
class Model:
    weights: jax.Array
    bias: float
    name: str

m = Model(weights=jax.numpy.array([1.0, 2.0]), bias=0.5, name="linear")
save("example.hdf5", m)
print(load("example.hdf5"))  # Model(weights=Array([1., 2.], dtype=float32), bias=0.5, name='linear')
```

Machine learning packages such as [Equinox](https://docs.kidger.site/equinox/)
automatically make all modules Python dataclasses, so Jaxon is fully compatible
with them.


#### Schema evolution

If a dataclass has changed between saving and loading (fields added or removed),
the following options control the behavior:

```python
result = load(
    "example.hdf5",
    allow_missing_fields=True,   # fields in file but absent from the class: warn and skip
    allow_unknown_fields=True,   # fields in class but absent from file: use default or an instance of JaxonNotLoaded
)
```


### Python numeric types

By default, Jaxon preserves the Python types `int`, `float`, `bool`, and `complex`
exactly. If `exact_python_numeric_types=False` is passed to `save`, they are
converted to their respective NumPy types `int_`, `double`, `bool_` and `cdouble`
which are stored using their corresponding native HDF5 types. This conversion
does not cause any data loss (in the case of integer overflows, the conversion is
silently skipped). When the HDF5 file is loaded, the numbers are NumPy scalars:

```python
save("data.hdf5", {"x": 1, "y": 3.14}, exact_python_numeric_types=False)
result = load("data.hdf5")
# result["x"] is np.int64(1), result["y"] is np.float64(3.14)
```

To convert only specific Python types, use `py_to_np_types`:

```python
save("data.hdf5", data, py_to_np_types=(int, float))  # int and float become int_ and double
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

### The `to_jaxon`/`from_jaxon` interface

For types that are not natively supported and are not dataclasses, the methods
`to_jaxon` and `from_jaxon` can be implemented:

```python
from jaxon import save, load
import numpy as np

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

If `to_jaxon` is provided, it takes priority over the implementation for dataclasses. Jaxon stores
the module and class name to identify the type.


### Custom marshaler/unmarshaler functions

For types that cannot be modified, callables can be passed to `save` and `load`:

```python
from jaxon import save, load
import numpy as np

class MyClass:
    def __init__(self, value):
        self.value = value

def marshal(obj):
    if isinstance(obj, MyClass):
        return "mymodule.MyClass", {"value": obj.value}
    return None  # signal that this marshaler cannot handle the type

def unmarshal(qualified_name, data):
    if qualified_name == "mymodule.MyClass":
        return MyClass(data["value"])
    return None

save("data.hdf5", MyClass(5), custom_marshalers=[marshal])
result = load("data.hdf5", custom_unmarshalers=[unmarshal])
```

If multiple marshalers are provided the first that returns a non-`None` result is used.
Note that the `qualified_name` returned by the marshaler can be an arbitrary identifier, but it must be unique among all marshalers (must not start with `dclass`,
`jaxon` or `dill`) and must not include the control characters `:`, `#` and `'`. Whitespace
at the end and start is stripped during loading. Typically, it is a string of the form
`marshaler_name[type_info]` where `marshaler_name` identifies the marshaler and `type_info`
is additional information for type reconstruction. Custom marshalers take priority over the
`to_jaxon`/`from_jaxon` and dataclass marshaling, which are internally also implemented
as marshaler functions. 


### Serialization with dill

If the type is not supported and all marshalers return `None`, Jaxon can use the
serializer [dill](https://dill.readthedocs.io/en/latest/) (an extended pickle) and
store the resulting binary blob. This is possibly unsafe and leads to a data
representation that is no longer human readable and self-describing. It must be
opted into explicitly:

```python
save("data.hdf5", obj, allow_dill=True)
result = load("data.hdf5", allow_dill=True)
```

Note that reference of objects elsewhere in the pytree that point to objects inside
the dill'ed object are broken after loading.


## Partial loading with `load_filter`

Large pytrees can be partially loaded by providing a filter function. The filter
receives the path to each node as a list and returns `True` to load it or `False`
to skip it (skipped nodes are replaced with `JAXON_NOT_LOADED`):

```python
from jaxon import load, save, has_common_prefix, JaxonNotLoaded
import numpy as np

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

# only load pytree.weights and anything nested under it
result = load("model.hdf5", load_filter=lambda path: has_common_prefix(["weights"], path))
print(result.weights)  # np.ndarray — loaded
print(result.config)   # JAXON_NOT_LOADED — skipped
```

The function `has_common_prefix(prefix, path)` returns `True` if `path` starts with `prefix`.
For dictionaries the path element is the dict key, for lists and sets it is the integer index,
and for dataclasses it is the field name. Dict keys themselves are always loaded regardless of
the filter. Note that sets are unordered; therefore the index of an element might change between
save cycles. This is an open issue which will be addressed in a future update.


## Schema

This section describes how Jaxon stores pytrees in an HDF5 file.


### Overall design principles

1. Jaxon adheres to the principle of storing data efficiently wherever the user provides the
data in an efficiently packed format (e.g. as an NumPy array). In other cases (e.g. for Python
lists of integers, lists of dataclasses, etc.) Jaxon prefers simplicity over efficiency.
2. Jaxon must, by default, restore every type and reference exactly as it was saved.
3. The resulting format must be simple, non-cryptic and self-describing (like JSON). In other
words, it must be understandable, read- and writeable without this library and its documentation.


### Introduction

To address the goal of simplicity, Jaxon represents each stored pytree leaf or node (e.g. dataclass
fields, list items, list objects themselves, ...) by a single HDF5 attribute. This also helps to keep
track of their order in lists. For container types, a group is created additionally which stores the
contents of the container. For arrays and buffers, a dataset is created which holds the data.



Consider the following example:

```python
from jaxon import save
import numpy as np

pytree = {
    "str_key": "value",
    "int_key": 1,
    "numpy_int": np.int_(3),
    "numpy_half": np.float16(3.2),
    "list_key": [1]
}
save("data.hdf5", pytree)
```
Running the script and then using h5dump to inspect the contents
```
h5dump data.hdf5 | grep -E '(GROUP|ATTRIBUTE|DATATYPE)'
```
outputs:
```
GROUP "/" {
   ATTRIBUTE "JAXON_ROOT:dict" {
      DATATYPE  H5T_IEEE_F64LE
   ATTRIBUTE "JAXON_VERSION" {
      DATATYPE  H5T_STRING {
   GROUP "JAXON_ROOT" {
      ATTRIBUTE "'int_key'" {
         DATATYPE  H5T_STRING {
      ATTRIBUTE "'list_key':list" {
         DATATYPE  H5T_IEEE_F64LE
      ATTRIBUTE "'numpy_half'" {
         DATATYPE  H5T_IEEE_F16LE
      ATTRIBUTE "'numpy_int'" {
         DATATYPE  H5T_STD_I64LE
      ATTRIBUTE "'str_key'" {
         DATATYPE  H5T_STRING {
      GROUP "'list_key'" {
         ATTRIBUTE "0" {
            DATATYPE  H5T_STRING {
```
The dictionary object itself is represented by the HDF5 attribute `JAXON_ROOT:dict`. Internally,
`JAXON_ROOT` is referred to as the attribute name and `dict` as the typehint. For a `dict`, Jaxon
expects a group with the name of the attribute, here visible as `GROUP "JAXON_ROOT"`. To ensure
compatibility with future versions, Jaxon also stores the version of the library in the special
`JAXON_VERSION` attribute. In the group `JAXON_ROOT`, it can be seen that for each entry in the
dict an attribute has been created. In simple cases (when the dictionary key can be represented
as a string without typehint), the dictionary key is used directly as the key in the HDF5 group.
This is the case in this example. Note that the single quotes come from jaxon. They ensure that
the `str` type can be told apart from a stringified `None`, `Ellipsis`, `bool`, `int`, `float`,
`complex` or `range`. For the two NumPy values, the type can be mapped directly to a native HDF5
type and therefore does not require a typehint. For the list another subgroup is created. The
indices are ignored during loading. Note that Jaxon always uses fixed length utf-8 encoded
strings.


### Python containers

####  dict

As elaborated in the introduction, each dict is represented by a group. If the dict keys can be
represented as strings without typehints, they are used directly as HDF5 group keys. Otherwise,
two attributes are created: One with the HDF5 group key `key(i)` which must be immediately followed
by an attribute with HDF5 group key `value(i)`. Here, `i` is the index of the entry in the dict, which is checked during loading. The two attributes store the key and value pytrees, respectively.

#### list, tuple, set, frozenset

Python `list`, `tuple`, `set` and `frozenset` objects are all stored in the same manner: A group
is created in which the HDF5 keys are plain (stringified) indices starting at zero. Note that
these indices are not checked during loading. 

### NumPy generic types

Jaxon supports all subtypes of `numpy.number` plus `bool_`. They are mapped to their
corresponding native HDF5 type and therefore stored compactly in binary form.
Furthermore, the following NumPy types are supported:

|NumPy dtype |HDF5 Type  |HDF5 Character Set |Jaxon typehint |
|------------|-----------|-------------------|---------------|
|bytes_      |H5T_STRING |H5T_CSET_ASCII     |numpy.bytes_   |
|str_        |H5T_STRING |H5T_CSET_UTF8      |numpy.str_     |
|void        |H5T_OPAQUE |                   |numpy.void     |

The data is always stored in an HDF5 dataset. Note that for `bytes_` and `str_` trailing null
bytes are removed. This is a standard practice.

### NumPy arrays

NumPy arrays are always stored in HDF5 datasets; the attribute has the typehint `numpy.ndarray`.
The array can have a scalar or structured datatype of the NumPy types listed above. However,
in structured arrays, `numpy.str_` is not supported as a field type. Also, titles are not
preserved (field names are always preserved). If a title is detected, a warning is emitted.


### Jax arrays

Jax arrays are internally converted to NumPy arrays before saving and converted back to jax
arrays during loading. The typehint is `jax.Array[...]` where `...` is the jax arrays's dtype.
It is mapped to a compatible NumPy datatype using the dictionary `JAXON_JAX_TO_NUMPY_TYPE`.
Note that this might increase the array size as neither NumPy nor HDF5 have native support
for certain (especially small) jax dtypes. This decision is a design tradeoff between
portability and efficiency. In future updates compression options are added, which will
increase efficiency.


### Python types

The following Python types are stored by converting them to strings. The representation
is mostly the same as produced by Python's `repr(...)` function.

|Python type |Encoding                                                |
|------------|--------------------------------------------------------|
|bool        |`repr(...)` of the bool                                 |
|int         |`repr(...)` of the int                                  |
|float       |`repr(...)` of the float                                |
|complex     |`repr(...)` of the complex; brackets are always removed |
|str         |raw string, quoted with single quotes ¹                |
|None        |`None`                                                  |
|Ellipsis    |`Ellipsis`                                              |
|range       |`repr(...)` of the range                                |

¹ Before storing, `'` and `\` characters are escaped using `\`.

The rational behind encoding these python datatypes as strings is that Python dictionaries map
most naturally to HDF5 groups, where the keys must be provided as strings.

The following Python types are supported by converting them to one of the
NumPy type above and adding the specified typehint:

|Python type |Converted to |Typehint  |
|------------|-------------|----------|
|`bytes`     |numpy.void   |bytes     |
|`bytearray` |numpy.void   |bytearray |


### References

If the same object is seen twice in the pytree, Jaxon inserts a reference to the object
on the second and following occurrences instead of storing a copy. A reference does not
have a typehint. The attribute value is a string (without quotation marks) of the form
`/JAXON_ROOT/group1/group2/.../attribute_name:typehint`. It always begins with
`/JAXON_ROOT/` followed by a path in the HDF5 file by specifying group names separated
by `/` character. The last path element is the full HDF5 key of the target attribute.


### Whitespace

All whitespace in the beginning or end of an HDF5 group key, attribute key or an attribute
value containing stringified data (like a Python integer, string or a reference) is ignored.
Furthermore, whitespace around the control characters `/`, `#`, `[`, `]` and `:` is ignored.


### Marshalers

If a type is not one of the above, marshalers are applied in order to convert it to one of the
above. Jaxon comes with a marshaler for dataclasses, one for objects that support the
`to_jaxon`/`from_jaxon` interface and one that uses dill. The latter one must be opted in
explicitly as explained earlier. When a marshaler is applied, jaxon appends a `#` to the typehint
followed by the type hint that the marshaler produced. For example, the dataclass marshaler
produces the typehint `dclass[...]` where `...` is the module and class name of the dataclass
separated by a dot. The jaxon interface marshaler produces `jaxon[...]`. Note that the marshaled
representation itself (for example it is a dict for dataclasses) can nowhere be referenced in the pytree (if it is referenced, the reference
will be broken). The returned, marshaled object is considered to be a temporary representation of
the object. Note that a marshaler might return any type, including one that must be processed
by another marshaler. In this case, another `#` is appended to the typehint forming a chain
of marshalers.


## Acknowledgements

Jaxon is built on the following libraries:

- [NumPy](https://numpy.org/)
- [JAX](https://github.com/jax-ml/jax)
- [h5py](https://www.h5py.org/)
- [dill](https://dill.readthedocs.io/en/latest/)
- [pytest](https://pytest.org)
