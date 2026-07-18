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
This module handles marshaling of a pytree to an intermediate
representation and storing it into the HDF5 file.

Author
------
Frank Hermann
"""


from importlib.metadata import version
from typing import Iterable, assert_never, Union
import warnings
from dataclasses import dataclass, is_dataclass, fields, field
import numpy as np
import h5py
import dill
from ._common import (
    JAXON_JAX_ARRAY, JAXON_NUMPY_ARRAY, JAXON_NUMPY_ARRAY_TYPES, JAXON_NUMPY_STR,
    JAXON_NUMPY_BYTES, JAXON_NUMPY_VOID, JAXON_PY_CONTAINER_TYPES, PyTree, Marshaler,
    JAXON_NUMPY_ATOMIC_TYPES, JaxonPyNumeric, JAXON_PY_NUMERIC_TYPES, JAXON_NUMPY_NUMERIC_TYPES,
    JAXON_NUMPY_NUMERIC_DTYPES, JAXON_JAX_ARRAY_TYPE, JAXON_NONE, JAXON_ELLIPSIS,
    JAXON_ROOT_GROUP_KEY, JAXON_DICT_KEY, JAXON_DICT_VALUE, JAXON_VERSION_GROUP_KEY,
    CircularPyTreeError, get_qualified_name, JAXON_JAX_TO_NUMPY_TYPE, JaxonNumpyAtomic,
    MIN_LENGTH_FOR_REFERENCEABLE_STR, JaxonTypeWarning, JaxonTypeError
)


@dataclass
class JaxonDict:
    """Internal representation of a dict."""
    data: list[tuple["JaxonAtom", "JaxonAtom"]] = field(default_factory=list)


@dataclass
class JaxonList:
    """Internal representation of a list, tuple, set or frozenset."""
    data: list["JaxonAtom"] = field(default_factory=list)


JaxonPrimitive = Union[JaxonNumpyAtomic, np.ndarray, str, JaxonList, JaxonDict, "JaxonAtom"]


@dataclass
class JaxonAtom:
    """
    Internal representation of any data item (including containers).
    
    Attributes
    ----------
    data: _JaxonPrimitive
        Encodes the pytree data, converted to a smaller subset of types.
    typehint: str
        Contains type information for pytree reconstruction. Will be stored
        in the HDF5 attribute key. The `typehint` can include marshaler type
        hints and type hints for the primitive type in `data`.
    original_obj: PyTree | None
        Holds a reference to the original pytree object.
    can_be_referenced: bool
        Indicates if it is allowed to reference the type in the jaxon file
        instead of storing copies.
    data_path: str
        After the atom has been stored, this string is set to the path.
    """
    data: JaxonPrimitive
    typehint: str = ""
    original_obj: PyTree | None = None
    can_be_referenced: bool = True
    data_path: str | None = None

    def __post_init__(self):
        assert type(self.data) in JAXON_PRIMITIVE_TYPES, "tried to construct _JaxonAtom " \
            "with invalid data type"
        assert self.typehint == self.typehint.strip(), "whitespace not allowed"

    def is_simple(self) -> bool:
        """A simple atom can be used as a group or attribute key in the HDF5 file
        and reconstructed from the key data alone. For this, the atom cannot have
        a typehint, `self.data` must be a `numpy.str_` that does not contain
        null chars and `self.can_be_referenced` must be `False`."""
        return (self.typehint == ""
            and type(self.data) is str
            and "\0" not in self.data
            and not self.can_be_referenced)


JAXON_PRIMITIVE_TYPES = JAXON_NUMPY_ATOMIC_TYPES | \
    {np.ndarray, str, JaxonList, JaxonDict, JaxonAtom}


def key_to_debugstring(dict_key: PyTree, i: int) -> str:
    if isinstance(dict_key, (str, int, float, bool, complex)):
        return repr(dict_key)
    return f"{i}"


def dataclass_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
    if not is_dataclass(pytree):
        return None
    th = get_qualified_name(pytree)
    res = {field.name: getattr(pytree, field.name) for field in fields(pytree)}
    return f"dclass[{th}]", res


def jaxon_interface_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
    if hasattr(pytree, "to_jaxon"):
        return f"jaxon[{get_qualified_name(pytree)}]", pytree.to_jaxon()
    return None


def supply_dill_marshaler(dill_kwargs) -> Marshaler:
    def dill_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
        return "dill", dill.dumps(pytree, **dill_kwargs)
    return dill_marshaler


def base_type(obj: PyTree, types: Iterable[type],
        downcast_to_base_types: frozenset[type]) -> None | type:
    """Check if the type `t=type(obj)` is in `types`. If `t` is
    not specified in `downcast_to_base_types`, `t` must be exactly equal
    to a type in `types`; otherwise `obj` must be an instance of a type in
    `types`. If a matching (base) type is found return it, otherwise return None."""
    if type(obj) in downcast_to_base_types:
        for t in types:
            if isinstance(obj, t):
                return t
    else:
        for t in types:
            if type(obj) is t:
                return t
    return None


def convert_py_numeric_to_numpy(pytree: JaxonPyNumeric) -> \
        np.bool_ | np.int_ | np.double | np.cdouble | None:
    if isinstance(pytree, bool):
        return np.bool_(pytree)
    if isinstance(pytree, int):
        try:
            return np.int_(pytree)
        except OverflowError:
            return None
    if isinstance(pytree, float):
        return np.double(pytree)
    if isinstance(pytree, complex):
        return np.cdouble(pytree)
    assert_never(pytree)  # pragma: no cover


def escape_str_terminator(string: str) -> str:
    return string.replace("\\", "\\\\").replace("'", "\\'")


def check_numpy_array_dtype(dtype: np.dtype, root_dtype: np.dtype, debug_path: str) -> None:
    if dtype.names is None:
        if dtype.type not in JAXON_NUMPY_ARRAY_TYPES:
            raise JaxonTypeError(f"dtype {root_dtype!r} of the numpy array at "
                                 f"{debug_path!r} is not supported")
        return
    for name in dtype.names:
        dtype_field = dtype.fields[name]
        check_numpy_array_dtype(dtype_field[0], root_dtype, debug_path)
        if len(dtype_field) > 2:
            warnings.warn(f"A field of dtype {root_dtype!r} of the numpy array at "
                          f"{debug_path!r} has a field name set which is unsupported by "
                          "jaxon. It will not be stored in the HDF5 file.", JaxonTypeWarning)


def convert_builtin_to_atom(pytree: PyTree,
        downcast_to_base_types: frozenset[type],
        py_to_np_types: frozenset[JaxonPyNumeric],
        marshalers: tuple[Marshaler, ...],
        parent_ids: frozenset[int],
        debug_path: str,
        cached_atoms: dict[int, JaxonAtom]) -> JaxonAtom | None:
    """Convert `pytree` to an atom if its a `JaxonBuiltin` or can be converted
    to a `JaxonBuiltin, return `None` otherwise."""
    # handle simple types which do not require references to be preserved
    if pytree is None:
        return JaxonAtom(JAXON_NONE, can_be_referenced=False)
    if pytree is Ellipsis:
        return JaxonAtom(JAXON_ELLIPSIS, can_be_referenced=False)
    py_numeric_type = base_type(pytree, JAXON_PY_NUMERIC_TYPES, downcast_to_base_types)
    if py_numeric_type is not None:
        if py_numeric_type in py_to_np_types:
            converted = convert_py_numeric_to_numpy(pytree)
            if converted is not None:
                # since we created the instance of the numpy generic
                # it can not have been referenced elsewhere
                return JaxonAtom(converted, can_be_referenced=False)
            # can not convert (because of int overflow)
        rep = repr(pytree)
        if py_numeric_type is complex:
            rep = rep[1:-1] if rep[0] == '(' else rep
        return JaxonAtom(rep, can_be_referenced=False)

    # handle python range
    if isinstance(pytree, (range,)):  # range cannot be subclassed;
                                      # so downcast_to_base_types is irrelevant
        # remove unnecessary spaces
        return JaxonAtom(repr(pytree).replace(" ", ""))

    # numpy generics are stored directly; without typehints
    # except for str_, bytes_ and void (these go into datasets later)
    #  - str_ needs a typehint to avoid ambiguity with stringified python data
    #  - bytes_ and void have typehint for consistency with str_, and because of
    #    the special handling for empty str_, bytes_ and void
    numpy_generic_type = base_type(pytree, JAXON_NUMPY_ATOMIC_TYPES, downcast_to_base_types)
    if numpy_generic_type is not None:
        if numpy_generic_type is np.str_:
            return JaxonAtom(pytree, JAXON_NUMPY_STR)
        if numpy_generic_type is np.bytes_:
            return JaxonAtom(pytree, JAXON_NUMPY_BYTES)
        if numpy_generic_type is np.void:
            return JaxonAtom(pytree, JAXON_NUMPY_VOID)
        return JaxonAtom(pytree)

    # handle python strings
    str_type = base_type(pytree, (str,), downcast_to_base_types)
    if str_type is not None:
        can_be_referenced = len(pytree) >= MIN_LENGTH_FOR_REFERENCEABLE_STR
        return JaxonAtom("'" + escape_str_terminator(pytree) + "'",
                         can_be_referenced=can_be_referenced)

    # handle jax arrays
    if base_type(pytree, (JAXON_JAX_ARRAY_TYPE,), downcast_to_base_types):
        dtype = pytree.dtype
        if dtype not in JAXON_NUMPY_NUMERIC_DTYPES:
            dtype = JAXON_JAX_TO_NUMPY_TYPE.get(dtype.name)
            if dtype is None:
                raise JaxonTypeError(f"dtype {pytree.dtype.name!r} of the jax array at "
                                     f"{debug_path!r} is unsupported")
        return JaxonAtom(np.asarray(pytree, dtype=dtype),
                          f"{JAXON_JAX_ARRAY}[{pytree.dtype.name}]")

    # handle numpy arrays
    if base_type(pytree, (np.ndarray,), downcast_to_base_types):
        check_numpy_array_dtype(pytree.dtype, pytree.dtype, debug_path)
        return JaxonAtom(pytree, JAXON_NUMPY_ARRAY)

    # handle python byte buffer objects
    byte_buffer_type = base_type(pytree, (bytes, bytearray), downcast_to_base_types)
    if byte_buffer_type is not None:
        # The intermediate conversion to bytes is needed to avoid creating an array.
        # For `bytes` and `bytearray`, trailing null bytes need to be preserved,
        # therefore store as np.void.
        return JaxonAtom(np.void(bytes(pytree)), byte_buffer_type.__name__)

    # handle containers
    container_type = base_type(pytree, JAXON_PY_CONTAINER_TYPES, downcast_to_base_types)
    if container_type is not None:
        if isinstance(pytree, dict):
            data = JaxonDict()
            for i, (dict_key, dict_value) in enumerate(pytree.items()):
                key_atom = to_atom(dict_key, downcast_to_base_types, py_to_np_types,
                    marshalers, parent_ids, f"{debug_path}.key({i})", cached_atoms)
                dbgstr = f"{debug_path}.{key_to_debugstring(dict_key, i)}"
                value_atom = to_atom(dict_value, downcast_to_base_types, py_to_np_types,
                    marshalers, parent_ids, dbgstr, cached_atoms)
                data.data.append((key_atom, value_atom))
        else:
            data = JaxonList()
            for i, item in enumerate(pytree):
                item_atom = to_atom(item, downcast_to_base_types, py_to_np_types,
                    marshalers, parent_ids, f"{debug_path}({i})", cached_atoms)
                data.data.append(item_atom)
        return JaxonAtom(data, container_type.__name__)

    # no know base type
    return None


def to_atom(pytree: PyTree,
            downcast_to_base_types: frozenset[type],
            py_to_np_types: frozenset[JaxonPyNumeric],
            marshalers: tuple[Marshaler, ...],
            parent_ids: frozenset[int],
            debug_path: str,
            cached_atoms: dict[int, JaxonAtom]) -> JaxonAtom:
    """Recursively convert `pytree` to the internal representation."""

    # circular reference protection
    if id(pytree) in parent_ids:
        raise CircularPyTreeError("detected circular reference "
                                  f"in pytree at {debug_path!r}")
    parent_ids |= {id(pytree)}

    # check if atom has been seen before
    # this is not just increasing performance,
    # it is essential to recover references
    atom = cached_atoms.get(id(pytree))
    if atom is not None:
        return JaxonAtom(atom, can_be_referenced=False)

    # marshal `pytree` till a builtin type is found
    marshaled = pytree
    typehint = ""
    while True:
        atom = convert_builtin_to_atom(marshaled, downcast_to_base_types, py_to_np_types,
            marshalers, parent_ids, debug_path, cached_atoms)
        if atom is not None:
            break
        result = None
        for marshaler in marshalers:
            result = marshaler(pytree)
            if result is not None:
                break
        if result is None:
            raise JaxonTypeError(f"Object at {debug_path!r} is not a valid jaxon type, but "
                "it can be serialized if allow_dill is set to True.")
        marshaled = result[1]
        typehint = "#" + result[0] + typehint
    atom.typehint += typehint  # add marshaler typehints
    atom.original_obj = pytree

    # register the original atom in the cache
    if atom.can_be_referenced:
        cached_atoms[id(pytree)] = atom
    return atom


def store_str_in_attrib(group: h5py.Group, attrib_key: str, value: str) -> None:
    assert len(value) > 0, "string should not have length 0"
    encoded = value.encode('utf-8')
    dt = h5py.string_dtype(encoding='utf-8', length=len(encoded))
    group.attrs.create(attrib_key, data=np.bytes_(encoded), dtype=dt)


def store_atom(group: h5py.Group, atom: JaxonAtom, attrib_name: str, group_path: str) -> None:
    """Recursively store the internal representation in the HDF5 file."""
    assert group_path[-1] == "/"
    if type(atom.data) is JaxonAtom:
        target_atom = atom.data
        assert target_atom.data_path is not None, "atom should have been visited before"
        store_str_in_attrib(group, attrib_name, target_atom.data_path)
        return

    # Decide what to store in the attribute data. Also, store children, if any.
    group_key = attrib_name  # if a sub group is generated
                             # it will have the same name as attrib_name
    sub_group_path = group_path + group_key + "/"
    if type(atom.data) is JaxonDict:
        sub_group = group.create_group(group_key, track_order=True)
        for i, (key_atom, value_atom) in enumerate(atom.data.data):
            if key_atom.is_simple():
                assert isinstance(key_atom.data, str)
                attrib_key_of_value = key_atom.data
            else:
                # If the dict key atom is not simple it cannot be used directly
                # as the attribute key in the HDF5 file. So it must be stored
                # using an extra attribute.
                attrib_key_of_value = f"{JAXON_DICT_VALUE}({i})"
                attrib_key_of_key = f"{JAXON_DICT_KEY}({i})"
                store_atom(sub_group, key_atom, attrib_key_of_key, sub_group_path)
            store_atom(sub_group, value_atom, attrib_key_of_value, sub_group_path)
        attrib_data = np.array([])
    elif type(atom.data) is JaxonList:
        sub_group = group.create_group(group_key, track_order=True)
        for i, item_atom in enumerate(atom.data.data):
            store_atom(sub_group, item_atom, str(i), sub_group_path)
        attrib_data = np.array([])
    elif type(atom.data) in (np.str_, np.bytes_, np.void):
        # zero length strings are not allowed, therefore they are stored with length 1
        # and marked with h5py.Empty ()
        if type(atom.data) is np.str_:
            data = atom.data.encode('utf-8')
            dtype = h5py.string_dtype(encoding='utf-8', length=max(len(data), 1))
            is_empty_str = len(data) == 0
        elif type(atom.data) is np.bytes_:
            data = atom.data
            dtype = h5py.string_dtype(encoding='ascii', length=max(len(data), 1))
            is_empty_str = len(data) == 0
        elif type(atom.data) is np.void:
            data = atom.data
            dtype = None
            is_empty_str = data.nbytes == 0
        else:
            assert False  # pragma: no cover
        if is_empty_str:
            data = h5py.Empty(dtype)
            dtype = None
        group.create_dataset(group_key, data=data, dtype=dtype)
        attrib_data = np.array([])
    elif type(atom.data) is np.ndarray:
        group.create_dataset(group_key, data=atom.data)
        attrib_data = np.array([])
    elif type(atom.data) in (JAXON_NUMPY_NUMERIC_TYPES | {str}):
        attrib_data = atom.data
    else:
        assert False, "invalid internal type encountered"  # pragma: no cover

    # build full attribute key
    atom.data_path = group_path + attrib_name
    attrib_key = attrib_name
    if atom.typehint:
        attrib_key += f":{atom.typehint}"

    # store attribute data
    if type(attrib_data) is str:
        store_str_in_attrib(group, attrib_key, attrib_data)
    else:
        group.attrs[attrib_key] = attrib_data


def save(path_or_file, pytree: PyTree,
         exact_python_numeric_types: bool = True,
         downcast_to_base_types: Iterable[type] | None = None,
         py_to_np_types: Iterable[JaxonPyNumeric] | None = None,
         allow_dill: bool = False,
         dill_kwargs: dict | None = None,
         custom_marshalers: Iterable[Marshaler] = tuple()) -> None:
    """
    Save a pytree in a human readable format in an HDF5 file with the specified path or
    write it to the provided file object. If the file already exists (or a file object is
    provided), it is truncated at the beginning.

    Parameters
    ----------
    path_or_file :
        A path-like object indicating the file path or a file-like object with the methods
        `read()` (or `readinto()`), `write()`, `seek()`, `tell()`, `truncate()`
        and `flush()`. Providing a path-like object is the preferred option if possible
        (see the h5py documentation).
    pytree : PyTree
        The pytree object to be saved. Can contain nested structures of arrays, lists,
        dicts, etc. (see README)
    exact_python_numeric_types : bool, default=True
        If `False`, the types `int`, `float`, `bool` and `complex` will be converted
        implicitly to `np.int_`, `np.double`, `np.bool_` and `np.cdouble` respectively
        and stored as the corresponding HDF5 binary type. If the file is loaded, the types will
        be the numpy (not python) types.
    downcast_to_base_types : Iterable[type]
        If a superclass of a supported base type is encountered in the pytree and the encountered
        type is contained in this Iterable, it is converted to and stored as the supported base type.
        This means that it is also reconstructed as this base type when the file is loaded.
    py_to_np_types : Iterable[JaxonPyNumeric]
        Apply the behavior of `exact_python_numeric_types` only to the python types in the given
        Iterable. If not `None`, `exact_python_numeric_types` will be ignored.
    allow_dill : bool, default=False
        Whether to allow `dill` as a fallback for serializing unsupported objects.
    dill_kwargs : dict or None, optional
        Keyword arguments passed to `dill.dumps` if `allow_dill` is `True`.
    custom_marshalers : Iterable[Marshaler]
        If provided, each custom node and leaf in the pytree (that has no jaxon builtin support) is
        passed to the Callables in the order they are provided. Each Callable shall return either
        `None` indicating that the Callable cannot marshal the type or a `tuple[str, PyTree]`
        representing a type hint (used for unmarshaling) and the corresponding marshaled object. It
        must either be another custom object or (typically) a jaxon builtin type such as a standard
        python container. If all Callables return `None` the object is marshaled using the
        `to_jaxon` interface (if available) or the default implementation for dataclasses. It is
        assumed that the object returned by the marshaler is nowhere referenced in the pytree.

    Raises
    ------
    TypeError
        Raised if an unsupported type or numpy dtype is found in the pytree.
    CircularPyTreeException
        Raised if a circular reference is detected in the pytree.

    Notes
    -----
    Please refer to the jaxon README to see the supported data types.
    """
    if py_to_np_types is None:
        if exact_python_numeric_types:
            py_to_np_types = frozenset()
        else:
            py_to_np_types = JAXON_PY_NUMERIC_TYPES
    py_to_np_types = frozenset(py_to_np_types)
    if downcast_to_base_types is None:
        downcast_to_base_types = frozenset()
    else:
        downcast_to_base_types = frozenset(downcast_to_base_types)
    if dill_kwargs is None:
        dill_kwargs = {}
    marshalers = list(custom_marshalers)
    marshalers += [jaxon_interface_marshaler, dataclass_marshaler]
    if allow_dill:
        marshalers.append(supply_dill_marshaler(dill_kwargs))
    root_atom = to_atom(pytree, downcast_to_base_types, py_to_np_types,
                        tuple(marshalers), frozenset(), "", {})
    if hasattr(path_or_file, "seek") and hasattr(path_or_file, "truncate"):
        # when a file like object is provided
        # the file must be truncated like this because the "w"
        # mode does not seem to do this (bug?)
        path_or_file.seek(0)
        path_or_file.truncate()
    with h5py.File(path_or_file, 'w', track_order=True) as file:
        store_str_in_attrib(file, JAXON_VERSION_GROUP_KEY, version('jaxon'))
        store_atom(file, root_atom, JAXON_ROOT_GROUP_KEY, "/")
