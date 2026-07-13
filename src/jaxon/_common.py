# Copyright (C) 2026  Frank Hermann

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""
Constants, type aliases, exception classes, internal data classes, and small
utility functions.

Author
------
Frank Hermann
"""


from types import EllipsisType
from typing import Any, Callable, Iterable
import numpy as np
import jax


# supported numpy numeric types
JAXON_NUMPY_INT_TYPES = {
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64
}
JAXON_NUMPY_FLOAT_TYPES = {np.float16, np.float32, np.float64}
if hasattr(np, "float128"):
    JAXON_NUMPY_FLOAT_TYPES.add(np.float128)
JAXON_NUMPY_COMPLEX_TYPES = {np.complex64, np.complex128}
if hasattr(np, "complex256"):
    JAXON_NUMPY_COMPLEX_TYPES.add(np.complex256)
JaxonNumpyNumeric = np.number | np.bool  # supported numpy numerical types
JAXON_NUMPY_NUMERIC_TYPES = (JAXON_NUMPY_INT_TYPES | JAXON_NUMPY_FLOAT_TYPES
                             | JAXON_NUMPY_COMPLEX_TYPES | {np.bool})
JAXON_NUMPY_NUMERIC_DTYPES = {np.dtype(t) for t in JAXON_NUMPY_NUMERIC_TYPES}

# supported numpy atomic types (`generic` types in numpy terminology)
JaxonNumpyAtomic = JaxonNumpyNumeric | np.str_ | np.bytes_ | np.void
JAXON_NUMPY_ATOMIC_TYPES = JAXON_NUMPY_NUMERIC_TYPES | {np.str_, np.bytes_, np.void}

# supported types for numpy arrays (these are not dtype objects)
JaxonNumpyArrayType = JaxonNumpyNumeric | np.bytes_ | np.void
JAXON_NUMPY_ARRAY_TYPES = JAXON_NUMPY_NUMERIC_TYPES | {np.bytes_, np.void}

# supported python numeric types
JaxonPyNumeric = int | float | bool | complex
JAXON_PY_NUMERIC_TYPES = {int, float, bool, complex}

# supported python atomic types
JaxonPyAtomic = JaxonPyNumeric | str | bytearray | bytes | None | EllipsisType | range
JAXON_PY_ATOMIC_TYPES = (JAXON_PY_NUMERIC_TYPES
                         | {str, bytearray, bytes, type(None), EllipsisType, range})

# all supported atomic types
JaxonAtomic = JaxonNumpyAtomic | JaxonPyAtomic
JAXON_ATOMIC_TYPES = JAXON_NUMPY_ATOMIC_TYPES | JAXON_PY_ATOMIC_TYPES

# supported python container types
JaxonPyContainer = list | tuple | dict | set | frozenset
JAXON_PY_CONTAINER_TYPES = {list, tuple, dict, set, frozenset}


# all supported builtin types (these are all supported types without custom types)
JaxonBuiltin = JaxonAtomic | JaxonPyContainer | np.ndarray | jax.Array


# type alias
PyTree = Any
PathElement = Any
Marshaler = Callable[[PyTree], tuple[str, PyTree] | None]
Unmarshaler = Callable[[str, PyTree], PyTree | None]
LoadFilter = Callable[[list[PathElement]], bool]
JAXON_JAX_ARRAY_TYPE = type(jax.numpy.array([]))  # get the type of a jax array
                                                   # (in a version-independent way)

# string shorter than this can never be referenced (copies are stored)
# string of equal or greater length are stored only once
MIN_LENGTH_FOR_REFERENCEABLE_STR = 15

# table to convert jaxon to numpy type
JAXON_JAX_TO_NUMPY_TYPE = {
    'uint2':              np.uint8,    # 2-bit unsigned  → uint8
    'uint4':              np.uint8,    # 4-bit unsigned  → uint8
    'int2':               np.int8,     # 2-bit signed    → int8
    'int4':               np.int8,     # 4-bit signed    → int8
    'float4_e2m1fn':      np.float16,  # 4-bit float     → float16
    'float8_e3m4':        np.float16,  # 8-bit float     → float16
    'float8_e4m3':        np.float16,  # 8-bit float     → float16
    'float8_e4m3fn':      np.float16,  # 8-bit float     → float16
    'float8_e4m3fnuz':    np.float16,  # 8-bit float     → float16
    'float8_e4m3b11fnuz': np.float16,  # 8-bit float     → float16
    'float8_e5m2':        np.float16,  # 8-bit float     → float16
    'float8_e5m2fnuz':    np.float16,  # 8-bit float     → float16
    'float8_e8m0fnu':     np.float16,  # 8-bit float     → float16
    # 16-bit bfloat   → float32 (not float16: different exponent range)
    'bfloat16':           np.float32,
}


# keywords which are used in the HDF5 file
JAXON_NONE = "None"  # used to encode python None
JAXON_ELLIPSIS = "Ellipsis"  # used to encode python Ellipsis (`...`)
JAXON_TRUE = "True"
JAXON_FALSE = "False"
JAXON_DICT_KEY = "key"  # used to indicate that this HDF5 attribute stores
                        # the key of another attribute in the same group
                        # (only used if necessary)
JAXON_DICT_VALUE = "value"  # used to indicate that this HDF5 attribute stores
                            # a dict value (only used iff `JAXON_DICT_KEY` is used)
JAXON_ROOT_GROUP_KEY = "JAXON_ROOT"  # HDF5 root group name (might be followed by typehint of
                                     # the root object)
JAXON_VERSION_GROUP_KEY = "JAXON_VERSION"
JAXON_JAX_ARRAY = "jax.Array"       # typehint for jax arrays
JAXON_NUMPY_ARRAY = "numpy.ndarray" # typehint for numpy arrays
JAXON_NUMPY_STR = "numpy.str_"      # typehint for numpy.str_
JAXON_NUMPY_BYTES = "numpy.bytes_"  # typehint for numpy.bytes_
JAXON_NUMPY_VOID = "numpy.void"     # typehint for numpy.void


class JaxonMissing:
    """Internal singleton flag object to indicate an unavailable result."""
    __slots__ = ()
JAXON_MISSING = JaxonMissing()


class DictKeyPathElement:
    """Internal singleton flag object to indicate that loader descended into a dict key."""
    __slots__ = ()
DICT_KEY_PATH_ELEMENT = DictKeyPathElement()


class JaxonWarning(UserWarning):
    """Base class for all warnings raised by Jaxon."""


class JaxonFormatWarning(JaxonWarning):
    """Warning that indicates an incompatible HDF5 file."""


class JaxonTypeWarning(JaxonWarning):
    """Warning that indicates an incompatible type in a pytree."""


class JaxonError(RuntimeError):
    """Base class for all errors raised by Jaxon."""


class JaxonFormatError(JaxonError):
    """Error that indicates an incompatible HDF5 file."""


class CircularPyTreeError(JaxonError):
    """Raised when a circular reference (reference to a parent object) was detected."""


class JaxonTypeError(JaxonError):
    """Error that indicates an incompatible type in a pytree."""


class JaxonNotLoaded:
    """Placeholder object to indicate data that has not been loaded."""
    __slots__ = ()

    def __repr__(self):
        return "JAXON_NOT_LOADED"

    def __eq__(self, other):
        """Supported so that this object is compatible with sets."""
        return False

    def __hash__(self):
        """Supported so that this object is compatible with sets."""
        return hash(type(self))


def has_common_prefix(path: Iterable, other_path: Iterable) -> bool:
    """Checks if the two Iterables start with the same values. If one of the Iterables
    is longer than the additional items are ignored."""
    return all(map(lambda ab: ab[0] == ab[1], zip(path, other_path)))


def get_qualified_name(obj) -> str:
    """The returned name fully identifies the class of the object so that a new object can be
    instantiated later during loading (see `_create_instance`)."""
    return type(obj).__module__ + "." + type(obj).__qualname__
