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
Pytree-to-atom conversion and HDF5 writing (save side).
"""


from typing import Any, Iterable
from dataclasses import is_dataclass, fields
import numpy as np
import h5py
import dill
from ._common import (
    JAXON_JAX_ARRAY, JAXON_NUMPY_ARRAY, JAXON_NUMPY_ARRAY_TYPES, JAXON_NUMPY_STR,
    JAXON_NUMPY_BYTES, JAXON_NUMPY_VOID, JAXON_PY_CONTAINER_TYPES, PyTree, Marshaler,
    JAXON_NUMPY_ATOMIC_TYPES, JaxonPyNumeric, JAXON_PY_NUMERIC_TYPES, JAXON_NUMPY_NUMERIC_TYPES,
    JAXON_NUMPY_NUMERIC_DTYPES, _JAXON_JAX_ARRAY_TYPE, JAXON_NONE, JAXON_ELLIPSIS,
    JAXON_ROOT_GROUP_KEY, JAXON_DICT_KEY, JAXON_DICT_VALUE, _JaxonAtom, _JaxonDict, _JaxonList,
    CircularPyTreeException, _get_qualified_name, _key_to_debugstring,
    _JAXON_JAX_TO_NUMPY_TYPE
)


def _dataclass_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
    if not is_dataclass(pytree):
        return None
    th = _get_qualified_name(pytree)
    res = {field.name: getattr(pytree, field.name) for field in fields(pytree)}
    return f"dclass[{th}]", res


def _jaxon_interface_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
    if hasattr(pytree, "to_jaxon"):
        return f"jaxon[{_get_qualified_name(pytree)}]", pytree.to_jaxon()
    return None


def _supply_dill_marshaler(dill_kwargs) -> Marshaler:
    def dill_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
        return "dill", dill.dumps(pytree, **dill_kwargs)
    return dill_marshaler


def _base_type(obj: Any, types: Iterable[type],
        downcast_to_base_types: tuple[type, ...]) -> None | type:
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


def _convert_py_numeric_to_numpy(pytree: JaxonPyNumeric) -> \
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
    assert False, "unreachable"


def _escape_str_terminator(string: str) -> str:
    return string.replace("\\", "\\\\").replace("'", "\\'")


def _convert_builtin_to_atom(pytree: PyTree, downcast_to_base_types: tuple[type, ...],
        py_to_np_types: set[JaxonPyNumeric], marshalers: tuple[Marshaler, ...],
        parent_ids: frozenset[int], debug_path: str,
        cached_atoms: dict[int, _JaxonAtom]) -> _JaxonAtom | None:
    """Convert `pytree` to an atom if its a `JaxonBuiltin` or can be converted to a `JaxonBuiltin,
    return `None` otherwise."""
    # handle simple types which do not require references to be preserved
    if pytree is None:
        return _JaxonAtom(JAXON_NONE, can_be_referenced=False)
    if pytree is Ellipsis:
        return _JaxonAtom(JAXON_ELLIPSIS, can_be_referenced=False)
    py_numeric_type = _base_type(pytree, JAXON_PY_NUMERIC_TYPES, downcast_to_base_types)
    if py_numeric_type is not None:
        if py_numeric_type in py_to_np_types:
            converted = _convert_py_numeric_to_numpy(pytree)
            if converted is not None:
                # since we created the instance of the numpy generic
                # it can not have been referenced elsewhere
                return _JaxonAtom(converted, can_be_referenced=False)
            # can not convert (because of int overflow)
        rep = repr(pytree)
        if py_numeric_type is complex:
            rep = rep[1:-1] if rep[0] == '(' else rep
        return _JaxonAtom(rep, can_be_referenced=False)

    # handle python range
    if isinstance(pytree, (range,)):  # range cannot be subclassed;
                                      # so downcast_to_base_types is irrelevant
        # remove unnecessary spaces
        return _JaxonAtom(repr(pytree).replace(" ", ""))

    # numpy generics are stored directly; without typehints
    # except for str_, bytes_ and void (these go into datasets later)
    numpy_generic_type = _base_type(pytree, JAXON_NUMPY_ATOMIC_TYPES, downcast_to_base_types)
    if numpy_generic_type is not None:
        if numpy_generic_type is np.str_:
            return _JaxonAtom(pytree, JAXON_NUMPY_STR)
        if numpy_generic_type is np.bytes_:
            return _JaxonAtom(pytree, JAXON_NUMPY_BYTES)
        if numpy_generic_type is np.void:
            return _JaxonAtom(pytree, JAXON_NUMPY_VOID)
        return _JaxonAtom(pytree)

    # handle python strings
    str_type = _base_type(pytree, (str,), downcast_to_base_types)
    if str_type is not None:
        return _JaxonAtom("'" + _escape_str_terminator(pytree) + "'")

    # handle jax arrays
    if _base_type(pytree, (_JAXON_JAX_ARRAY_TYPE,), downcast_to_base_types):
        dtype = pytree.dtype
        if dtype not in JAXON_NUMPY_NUMERIC_DTYPES:
            try:
                dtype = _JAXON_JAX_TO_NUMPY_TYPE[dtype.name]
            except KeyError:
                raise TypeError(f"unknown dtype {dtype.name!r}, of jax array at {debug_path!r}")
        return _JaxonAtom(np.asarray(pytree, dtype=dtype),
                          f"{JAXON_JAX_ARRAY}[{pytree.dtype.name}]")

    # handle numpy arrays
    if _base_type(pytree, (np.ndarray,), downcast_to_base_types):
        scalar_type = pytree.dtype.type
        if scalar_type not in JAXON_NUMPY_ARRAY_TYPES:
            raise TypeError(f"dtype {scalar_type!r} of numpy array at "
                            f"{debug_path!r} is not supported")
        return _JaxonAtom(pytree, JAXON_NUMPY_ARRAY)

    # handle python byte buffer objects
    byte_buffer_type = _base_type(pytree, (bytes, bytearray), downcast_to_base_types)
    if byte_buffer_type is not None:
        return _JaxonAtom(np.void(pytree), byte_buffer_type.__name__)

    # handle containers
    container_type = _base_type(pytree, JAXON_PY_CONTAINER_TYPES, downcast_to_base_types)
    if container_type is not None:
        if isinstance(pytree, dict):
            data = _JaxonDict()
            for i, (dict_key, dict_value) in enumerate(pytree.items()):
                key_atom = _to_atom(dict_key, downcast_to_base_types, py_to_np_types,
                    marshalers, parent_ids, f"{debug_path}.key({i})", cached_atoms)
                dbgstr = f"{debug_path}.{_key_to_debugstring(dict_key, i)}"
                value_atom = _to_atom(dict_value, downcast_to_base_types, py_to_np_types,
                    marshalers, parent_ids, dbgstr, cached_atoms)
                data.data.append((key_atom, value_atom))
        else:
            data = _JaxonList()
            for i, item in enumerate(pytree):
                item_atom = _to_atom(item, downcast_to_base_types, py_to_np_types,
                    marshalers, parent_ids, f"{debug_path}({i})", cached_atoms)
                data.data.append(item_atom)
        return _JaxonAtom(data, container_type.__name__)

    # no know base type
    return None


def _to_atom(pytree: PyTree, downcast_to_base_types: tuple,
             py_to_np_types: set[JaxonPyNumeric], marshalers: tuple[Marshaler, ...],
             parent_ids: frozenset[int], debug_path: str,
             cached_atoms: dict[int, _JaxonAtom]) -> _JaxonAtom:
    """
    Recursively convert `pytree` to the internal representation. This function handles:
        - caching (enables correct reconstruction of references during loading)
        - prevents infinite recursion (by detecting circular references)
        - marshaling of non-builtin types (marshalers can be chained)
    """
    # marshal `pytree` till a builtin type is found, or till a
    # pytree is found that has been seen before
    atom = None
    intermediates = []
    while True:
        atom = cached_atoms.get(id(pytree))
        if atom is not None:
            break
        if id(pytree) in parent_ids:
            raise CircularPyTreeException("detected circular reference "
                                          f"in pytree at {debug_path!r}")
        parent_ids |= {id(pytree)}
        atom = _convert_builtin_to_atom(pytree, downcast_to_base_types, py_to_np_types,
            marshalers, parent_ids, debug_path, cached_atoms)
        if atom is not None:
            # register the primitive atom
            if atom.can_be_referenced:
                atom.original_obj = pytree
                cached_atoms[id(pytree)] = atom
            break
        result = None
        for marshaler in marshalers:
            result = marshaler(pytree)
            if result is not None:
                break
        if result is None:
            raise TypeError(f"Object at {debug_path!r} is not a valid jaxon type, but it can be "
                             "serialized if allow_dill is set to True.")
        marshaler_typehint, marshaled = result
        intermediates.append((marshaler_typehint, pytree))
        pytree = marshaled

    # register the intermediates and built a linked list of atoms
    # that models the chain of marshalers
    for marshaler_typehint, intermediate_pytree in reversed(intermediates):
        # keep a reference to pytree, otherwise it might be deleted from
        # memory and reused (if its an intermediate result), making the
        # id() in cached_atoms invalid
        atom = _JaxonAtom(atom, marshaler_typehint, pytree)
        cached_atoms[id(intermediate_pytree)] = atom
    return atom


def _store_str_in_attrib(group, attrib_key_with_th: str, s: str) -> None:
    assert len(s) > 0, "string should not have length 0"
    encoded = s.encode('utf-8')
    dt = h5py.string_dtype(encoding='utf-8', length=len(encoded))
    group.attrs.create(attrib_key_with_th, data=np.bytes_(encoded), dtype=dt)


def _store_atom(group, atom: _JaxonAtom, attrib_name: str, group_path: str) -> None:
    """Recursively store the internal representation in the HDF5 file."""
    assert group_path[-1] == "/"

    # Follow references till an atom is found that was already
    # stored or can be stored now.
    reference_atoms = []
    while type(atom.data) is _JaxonAtom and atom.data_path is None:
        reference_atoms.append(atom)
        atom = atom.data

    # Decide what to store in the attribute data. Also, store children, if any.
    group_key = attrib_name  # if a sub group is generated
                             # it will have the same name as attrib_name
    sub_group_path = group_path + group_key + "/"
    is_reference = False
    if atom.data_path is not None:
        # If an atom was found that has already been stored, a reference to it must be
        # stored. Otherwise, the load side will not correctly recover the references
        # in the pytree.
        is_reference = True
        attrib_data = atom.data_path
    elif type(atom.data) is _JaxonDict:
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
                _store_atom(sub_group, key_atom, attrib_key_of_key, sub_group_path)
            _store_atom(sub_group, value_atom, attrib_key_of_value, sub_group_path)
        attrib_data = np.array([])
    elif type(atom.data) is _JaxonList:
        sub_group = group.create_group(group_key, track_order=True)
        for i, item_atom in enumerate(atom.data.data):
            _store_atom(sub_group, item_atom, str(i), sub_group_path)
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
        else:
            data = atom.data
            dtype = None
            is_empty_str = data.nbytes == 0
        # uncommenting this preserves trailing null bytes
        #if b'\0' in data and type(data) is not np.void:
        #    data = np.void(data)
        #    dtype = None
        if is_empty_str:
            data = h5py.Empty(dtype)
            dtype = None
        group.create_dataset(group_key, data=data, dtype=dtype)
        attrib_data = np.array([])
    elif type(atom.data) in (np.ndarray,):
        group.create_dataset(group_key, data=atom.data)
        attrib_data = np.array([])
    elif type(atom.data) in (JAXON_NUMPY_NUMERIC_TYPES | {str}):
        attrib_data = atom.data
    else:
        assert False, f"invalid internal type {type(atom.data).__name__!r} encountered"

    # start building full attribute key
    attrib_key = attrib_name
    # if `atom`` should be referenced, don't put its typehint (references never have typehints)
    has_typeinfo = atom.typehint and not is_reference
    if has_typeinfo:
        attrib_key += f":{atom.typehint}"

    # set paths so that atoms which are stored later can reference the data
    if not is_reference:
        # do not change data_path of referenced atom
        assert atom.data_path is None
        atom.data_path = group_path + attrib_key
    if reference_atoms and not has_typeinfo:
        # add colon if necessary and not already present
        attrib_key += ":"
    for reference_atom in reference_atoms[::-1]:
        attrib_key += "#" + reference_atom.typehint
        assert reference_atom.data_path is None
        reference_atom.data_path = group_path + attrib_key

    # store attribute data
    if type(attrib_data) is str:
        _store_str_in_attrib(group, attrib_key, attrib_data)
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
        ``read()`` (or ``readinto()``), ``write()``, ``seek()``, ``tell()``, ``truncate()``
        and ``flush()``. Providing a path-like object is the preferred option if possible
        (see the h5py documentation).
    pytree :
        The pytree object to be saved. Can contain nested structures of arrays, lists,
        dicts, etc. (see README)
    exact_python_numeric_types : bool, default=True
        If ``False``, the types ``int``, ``float``, ``bool`` and ``complex`` will be converted
        implicitly to ``np.int64``, ``np.float64``, ``np.bool_`` and ``np.complex128`` respectively
        and stored as the corresponding HDF5 binary type. If the file is loaded, the types will
        be the numpy (not python) types.
    downcast_to_base_types : Iterable[type]
        If a superclass of a supported base type is encountered in the pytree and is contained in
        this Iterable, it is converted to and stored as the supported base type. This means that
        it is also reconstructed as the supported base type when the file is loaded.
    py_to_np_types : Iterable[JaxonPyNumeric]
        Apply the behavior of ``exact_python_numeric_types`` only to the python types in the given
        Iterable. If not ``None``, ``exact_python_numeric_types`` will be ignored.
    allow_dill : bool, default=False
        Whether to allow ``dill`` for serializing unsupported objects.
    dill_kwargs : dict or None, optional
        Extra keyword arguments passed to ``dill.dumps`` if ``allow_dill`` is True.
    custom_marshalers : Iterable[Marshaler]
        If provided, each custom node in the pytree (that has not a builtin type) is passed to
        the Callables in the order they are provided. Each Callable shall return either ``None``
        indicating that the Callable cannot marshal the type or a ``tuple[str, PyTree]``
        representing the qualified type name (that is used for unmarshaling) and the
        corresponding marshaled object which must be another custom object or (typically) a
        python standard container type. If all Callables return ``None`` the object is
        marshaled using the ``to_jaxon`` interface (if available) or the default
        implementation for dataclasses.

    Notes
    -----
    - Please refer to the jaxon README to see the supported data types.
    """
    if py_to_np_types is None:
        if exact_python_numeric_types:
            py_to_np_types = set()
        else:
            py_to_np_types = JAXON_PY_NUMERIC_TYPES
    else:
        py_to_np_types = set(py_to_np_types)
    if downcast_to_base_types is None:
        downcast_to_base_types = tuple()
    else:
        downcast_to_base_types = tuple(downcast_to_base_types)
    if dill_kwargs is None:
        dill_kwargs = {}
    marshalers = list(custom_marshalers)
    marshalers += [_jaxon_interface_marshaler, _dataclass_marshaler]
    if allow_dill:
        marshalers.append(_supply_dill_marshaler(dill_kwargs))
    root_atom = _to_atom(pytree, downcast_to_base_types, py_to_np_types,
                         tuple(marshalers), frozenset(), "", {})
    if hasattr(path_or_file, "seek") and hasattr(path_or_file, "truncate"):
        # when a file like object is provided
        # the file must be truncated like this because the "w"
        # mode does not seem to do this (bug?)
        path_or_file.seek(0)
        path_or_file.truncate()
    with h5py.File(path_or_file, 'w', track_order=True) as file:
        _store_atom(file, root_atom, JAXON_ROOT_GROUP_KEY, "/")
