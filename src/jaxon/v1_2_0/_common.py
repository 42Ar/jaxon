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
utility functions shared by both the save and load modules.

Author
------
Frank Hermann
"""


from typing import Any, Callable, Iterable
from dataclasses import dataclass
import dataclasses
import numpy as np
import jax


# note that the following lists of types do not represent what is supported by jaxon
# (refer to the README)
JAXON_NP_NUMERIC_TYPE_NAMES = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "float16", "float32", "float64", "longdouble", "complex64", "complex128",
    "clongdouble", "bool_")  # canonical numpy scalar type names; all exist on every platform
JAXON_NP_NUMERIC_TYPES = tuple(getattr(np, typename) for typename in JAXON_NP_NUMERIC_TYPE_NAMES)
JAXON_PY_NUMERIC_TYPES = (int, float, bool, complex)  # supported python numeric types
JAXON_CONTAINER_TYPES = (list, tuple, dict, set, frozenset)  # supported python container types


# get the type of a jax array (in a version-independent way)
# it is used to detect jax arrays
JAXON_JAX_ARRAY_TYPE = type(jax.numpy.array([]))


# the following are keywords which are used in the HDF5 file
JAXON_NONE = "None"  # used to encode python `None`
JAXON_ELLIPSIS = "Ellipsis"  # used to encode python `...`
JAXON_DICT_KEY = "key"  # used to indicate that this HDF5 attribute stores
                        # the key of another attribute in the same group
                        # (only used if necessary)
JAXON_DICT_VALUE = "value" # used to indicate that this HDF5 attribute stores
                           # a dict value (only used iff `JAXON_DICT_KEY` is used)
JAXON_ROOT_GROUP_KEY = "JAXON_ROOT"  # HDF5 root group name (might be followed by typehint of
                                     # the root object)
JAXON_REF = "ref"  # typehint that indicates a path to another object in the
                   # hdf5 file which maps to a python reference


# type definitions
PyTree = Any
_PathElement = Any
Marshaler = Callable[[PyTree], tuple[str, PyTree] | None]
Unmarshaler = Callable[[str, PyTree], PyTree | None]
LoadFilter = Callable[[list[_PathElement]], bool]
_JaxonMissing = object


class JaxonFormatWarning(UserWarning):
    """Warning that indicates an incompatible hdf5 file"""


class JaxonError(RuntimeError):
    """Base class for all errors raised by Jaxon"""


class CircularPyTreeException(JaxonError):
    """Raised when a circular reference (reference to a parent object) was detected."""


@dataclass(frozen=True)
class _JaxonLoadedFromReferenceWrapper:
    """Indicates that the wrapped object has been loaded from a reference."""
    pytree: PyTree


class _JaxonNotLoadedType:
    """Type of the ``JAXON_NOT_LOADED`` sentinel. Not intended to be instantiated directly."""
    __slots__ = ()

    def __repr__(self):
        return "JAXON_NOT_LOADED"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(type(self))


class _DictKeyPathElement:
    """Flag object path element to indicate that loader descended into a dict key."""

    def __repr__(self):
        return "_DICT_KEY_PATH_ELEMENT"


JAXON_NOT_LOADED = _JaxonNotLoadedType()
JaxonNotLoaded = _JaxonNotLoadedType  # public alias; use for isinstance checks
_DICT_KEY_PATH_ELEMENT = _DictKeyPathElement()
_JAXON_MISSING = _JaxonMissing()


@dataclass
class JaxonDict:
    """Internal representation of a dict."""
    data: list[tuple['JaxonAtom', 'JaxonAtom']] = dataclasses.field(default_factory=list)


@dataclass
class JaxonList:
    """Internal representation of a list."""
    data: list['JaxonAtom'] = dataclasses.field(default_factory=list)


@dataclass
class JaxonAtom:
    """Internal representation of any data item (including containers). The `data`
    field encodes the actual data which has been converted to a smaller subset
    of possible types, which are `JAXON_NP_NUMERIC_TYPES`, `memoryview`, `np.ndarray`,
    `str` and if python to numpy type conversion is activated, also `JAXON_PY_NUMERIC_TYPES`.
    For certain types it is necessary to have an additional `typehint` to reconstruct the
    original type of `data`. The field `original_obj_id` keeps track of the `id(...)` of
    the pytree object that is or was converted to `data`."""
    data: Any
    typehint: str | None = None
    original_obj_id: int | None = None

    def is_simple(self) -> bool:
        """A simple atom encodes the data and typehint only into in the data field
        which must be a str that does not contain null chars. This means that
        simple atoms can be used as group or attribute keys in the HDF5 file."""
        return self.typehint is None and type(self.data) is str and "\0" not in self.data


@dataclass
class JaxonStorageHints:
    """If the field `store_in_dataset` is `True` the associated data will be stored in an HDF5
    dataset. Otherwise, it will be stored in an HDF5 attribute."""
    store_in_dataset: bool


def has_common_prefix(path: Iterable, other_path: Iterable) -> bool:
    """Checks if the two Iterables start with the same values. If one of the Iterables
    is longer than the additional items are ignored."""
    return all(map(lambda ab: ab[0] == ab[1], zip(path, other_path)))


def _get_qualified_name(obj) -> str:
    """The returned name fully identifies the class of the object so that a new object can be
    instantiated later during loading (see `_create_instance`)."""
    return type(obj).__module__ + "." + type(obj).__qualname__


def _key_to_debugstring(dict_key, i) -> str:
    if isinstance(dict_key, (str, int, float, bool, complex)):
        return repr(dict_key)
    return f"{i}"
