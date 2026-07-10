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
Public API for jaxon: re-exports save(), load(), and all public symbols.

Author
------
Frank Hermann
"""


from ._common import (
    JaxonNumpyNumeric,
    JAXON_NUMPY_NUMERIC_TYPES,
    JaxonNumpyAtomic,
    JAXON_NUMPY_ATOMIC_TYPES,
    JaxonNumpyArrayType,
    JAXON_NUMPY_ARRAY_TYPES,
    JaxonPyNumeric,
    JAXON_PY_NUMERIC_TYPES,
    JaxonPyAtomic,
    JAXON_PY_ATOMIC_TYPES,
    JaxonAtomic,
    JAXON_ATOMIC_TYPES,
    JaxonPyContainer,
    JAXON_PY_CONTAINER_TYPES,
    JaxonBuiltin,
    PyTree,
    Marshaler,
    Unmarshaler,
    LoadFilter,
    JAXON_NONE,
    JAXON_ELLIPSIS,
    JAXON_TRUE,
    JAXON_FALSE,
    JAXON_DICT_KEY,
    JAXON_DICT_VALUE,
    JAXON_ROOT_GROUP_KEY,
    JAXON_JAX_ARRAY,
    JAXON_NUMPY_ARRAY,
    JAXON_NUMPY_STR,
    JAXON_NUMPY_BYTES,
    JAXON_NUMPY_VOID,
    JaxonFormatWarning,
    JaxonError,
    CircularPyTreeException,
    JaxonNotLoaded,
    has_common_prefix
)
from ._save import save
from ._load import load

__all__ = [
    "save",
    "load",
    "JaxonNumpyNumeric",
    "JAXON_NUMPY_NUMERIC_TYPES",
    "JaxonNumpyAtomic",
    "JAXON_NUMPY_ATOMIC_TYPES",
    "JaxonNumpyArrayType",
    "JAXON_NUMPY_ARRAY_TYPES",
    "JaxonPyNumeric",
    "JAXON_PY_NUMERIC_TYPES",
    "JaxonPyAtomic",
    "JAXON_PY_ATOMIC_TYPES",
    "JaxonAtomic",
    "JAXON_ATOMIC_TYPES",
    "JaxonPyContainer",
    "JAXON_PY_CONTAINER_TYPES",
    "JaxonBuiltin",
    "PyTree",
    "Marshaler",
    "Unmarshaler",
    "LoadFilter",
    "JAXON_NONE",
    "JAXON_ELLIPSIS",
    "JAXON_TRUE",
    "JAXON_FALSE",
    "JAXON_DICT_KEY",
    "JAXON_DICT_VALUE",
    "JAXON_ROOT_GROUP_KEY",
    "JAXON_JAX_ARRAY",
    "JAXON_NUMPY_ARRAY",
    "JAXON_NUMPY_STR",
    "JAXON_NUMPY_BYTES",
    "JAXON_NUMPY_VOID",
    "JaxonFormatWarning",
    "JaxonError",
    "CircularPyTreeException",
    "JaxonNotLoaded",
    "has_common_prefix"
]
