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
    JAXON_NP_NUMERIC_TYPE_NAMES,
    JAXON_NP_NUMERIC_TYPES,
    JAXON_PY_NUMERIC_TYPES,
    JAXON_CONTAINER_TYPES,
    JAXON_JAX_ARRAY_TYPE,
    JAXON_ROOT_GROUP_KEY,
    PyTree,
    Marshaler,
    Unmarshaler,
    LoadFilter,
    JaxonFormatWarning,
    JaxonError,
    CircularPyTreeException,
    JaxonAtom,
    JaxonDict,
    JaxonList,
    JaxonStorageHints,
    JaxonNotLoaded,
    JAXON_NOT_LOADED,
    has_common_prefix,
)
from ._save import save
from ._load import load

__all__ = [
    # core API
    "save",
    "load",
    # exceptions and warnings
    "JaxonError",
    "JaxonFormatWarning",
    "CircularPyTreeException",
    # user-facing types
    "JaxonStorageHints",
    "JaxonNotLoaded",
    "JAXON_NOT_LOADED",
    # type aliases (useful for annotations)
    "PyTree",
    "Marshaler",
    "Unmarshaler",
    "LoadFilter",
    # public constants
    "JAXON_NP_NUMERIC_TYPE_NAMES",
    "JAXON_NP_NUMERIC_TYPES",
    "JAXON_PY_NUMERIC_TYPES",
    "JAXON_CONTAINER_TYPES",
    "JAXON_JAX_ARRAY_TYPE",
    "JAXON_ROOT_GROUP_KEY",
    # utilities
    "has_common_prefix",
]
