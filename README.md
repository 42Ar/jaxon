# Jaxon
Jaxon is a python library that implements saving and loading of pytrees
to the Hierarchical Data Format [HDF5](https://wikipedia.org/wiki/Hierarchical_Data_Format).
HDF5 is an open format that natively supports multidimensional array objects and metadata
information in a single file, resulting in high efficiency. Jaxon embeds all
information that is necessary to reconstruct the pytree in a human-readable and
self-describing way, so that the output file can still be understood
even when the original code is no longer or available, or when it is desired to
process the data wth an external tool.

Jaxon is well suited for machine learning or scientific tasks. Its is 
especially suited for machine learning packages that rely on Python Dataclasses and [JAX](https://github.com/jax-ml/jax), e.g.
[Equinox](https://docs.kidger.site/equinox/).


## Installation
```bash
pip install jaxon
```


## Example Usage
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
Will produce
```
{'mylist': ['foo', 'bar', 42], 'myset': {'z', 'a', 'b', (42, b'binary!')}, 'numpy_array': array([0, 1, 2]), 'jax_array': Array([0, 1, 2], dtype=int32)}
```
which is exactly what was send in. Refer to the `tests` folder for more examples.
To inspect the HDF5 file, an external tool like `h5dump` or `HDFView` can bes used.


## Supported Types
The `pytree` can consist of the following types

| Dataype                               | Stored As                                               |
| ------------------------------------- | ------------------------------------------------------- |
| list, tuple, dict, set, frozenset     | HD5F Group                                              |
| np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64, np.float128, np.complex64, np.complex128, np.bool | HD5F Attribute |
| int, float, bool, complex             | String representation, or one of the numpy types above if requested |
| None, slice, range, Ellipsis          | String representation
| str                                   | HD5F UTF-8 (fixed length) string                        |
| np.ndarray, jax.Array, bytes, bytearray, memoryview | HD5F Attribute (or Dataset on User Request) |
| Any Python Dataclass                  | HD5F Group, that contains all Fields                    |

Note that dictionary keys can also be of any of these types or a custom type (if its hashable, of course).

### Custom Types: Dataclasses
The most straightforward way to add custom types is to make them a python Dataclass. The package
name, the class name and all fields, including the field names are saved. During loading,
the class is instantiated (without calling `__init__`) and the field values are set
(even if the datalcass is frozen). Note that machine learning packages like
[Equinox](https://docs.kidger.site/equinox/) make all modules automatically a python
Dataclass. Therefore, Jaxon is fully compatible with models implemented with this package.


### Custom Types: The `to_jaxon` and `from_jaxon` methods
If during saving a type in the pytree is encountered that is not in the table above, jaxon first
checks if it has the `to_jaxon` method. If yes, it is ignored if the type is dataclass or
not. The `to_jaxon` method is called and it must return a supported python container or another
custom object. Jaxon remembers the package and class name. During loading, jaxon instantiates
the class (without calling `__init__`) and then calls the `from_jaxon` method to
initialize the class with the object that was returned during saving from the `to_jaxon` method.


### Custom Types: Serialization with dill
As a last resort, Jaxon can Serialize unsupported types using the `dill` library (basically an
enhanced pickle) and store the result as a binary blob. This feature must be enabled by setting
`allow_dill=True`. Note that human readability (through HD5F viewer) is lost. 


## Acknowledgements
Jaxon is build on the following amazing libraries.

  - [NumPy](https://numpy.org/)
  - [JAX](https://github.com/jax-ml/jax)
  - [h5py](https://www.h5py.org/)
  - [dill](https://dill.readthedocs.io/en/latest/)

The author expresses gratitude to the contributers of the open source community.
