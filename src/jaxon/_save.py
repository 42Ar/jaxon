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
from dataclasses import replace, is_dataclass, fields
import numpy as np
import h5py
import dill
from ._common import (
    JAXON_JAX_ARRAY, JAXON_NUMPY_ARRAY, JAXON_NUMPY_ARRAY_TYPES, JAXON_NUMPY_STR,
    JAXON_NUMPY_BYTES, JAXON_NUMPY_VOID, JAXON_PY_CONTAINER_TYPES, PyTree, Marshaler,
    JAXON_NUMPY_ATOMIC_TYPES, JaxonPyNumeric, JAXON_PY_NUMERIC_TYPES, JAXON_NUMPY_NUMERIC_TYPES,
    JAXON_NUMPY_NUMERIC_DTYPES, _JAXON_JAX_ARRAY_TYPE, _JaxonMissing, JAXON_NONE, JAXON_ELLIPSIS,
    JAXON_ROOT_GROUP_KEY, JAXON_DICT_KEY, JAXON_DICT_VALUE, _JaxonAtom, _JaxonDict, _JaxonList,
    CircularPyTreeException, _JAXON_MISSING, _get_qualified_name, _key_to_debugstring,
    _JAXON_JAX_TO_NUMPY_TYPE
)


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


def _marshal_dataclass(instance):
    return {field.name: getattr(instance, field.name) for field in fields(instance)}


def _marshal_custom_obj(pytree: PyTree, custom_marshalers: tuple[Marshaler, ...]) \
        -> tuple[str, PyTree] | None:
    """Opposite of ``_unmarshal_custom_obj``. Returns ``None`` if marshaling of the custom object
    is not supported, otherwise returns a tuple of the qualified name and the marshaled PyTree."""
    for custom_marshaler in custom_marshalers:
        result = custom_marshaler(pytree)
        if result is not None:
            return result
    if hasattr(pytree, "to_jaxon"):
        return _get_qualified_name(pytree), pytree.to_jaxon()
    if is_dataclass(pytree):
        return _get_qualified_name(pytree), _marshal_dataclass(pytree)
    return None


def _to_atom(pytree: PyTree, allow_dill: bool, dill_kwargs: dict, downcast_to_base_types: tuple,
             py_to_np_types: set[JaxonPyNumeric], custom_marshalers: tuple[Marshaler, ...],
             parent_ids: frozenset[int], debug_path: str,
             cached_atoms: dict[int, _JaxonAtom]) -> _JaxonAtom:
    """Recursively convert ``pytree`` to the internal representation. This function handles caching,
    which also enables correct reconstruction of references during loading. Also, this function
    prevents infinite recursion (by detecting circular references) and adds the original object
    id to the atom."""
    atom = _to_atom_non_reference_type(pytree, downcast_to_base_types, py_to_np_types)
    if atom is not _JAXON_MISSING:
        return replace(atom, original_obj_id=id(pytree))  # type: ignore (cannot be JaxonMissing)
    # from here, the data items can be bigger and it is worthwhile to cache them
    result = cached_atoms.get(id(pytree), _JAXON_MISSING)
    if result is not _JAXON_MISSING:
        return result  # type: ignore (cannot be JaxonMissing)
    if id(pytree) in parent_ids:
        raise CircularPyTreeException(f"detected circular reference in pytree at {debug_path!r}")
    parent_ids = parent_ids | {id(pytree)}
    atom = _to_atom_reference_type(pytree, allow_dill, dill_kwargs, downcast_to_base_types,
        py_to_np_types, custom_marshalers, parent_ids, debug_path, cached_atoms)
    atom = replace(atom, original_obj_id=id(pytree))
    cached_atoms[id(pytree)] = atom
    return atom


def _convert_py_numeric_to_numpy(pytree: JaxonPyNumeric) -> \
        np.bool_ | np.int_ | np.double | np.cdouble | _JaxonMissing:
    if isinstance(pytree, bool):
        return np.bool_(pytree)
    if isinstance(pytree, int):
        try:
            return np.int_(pytree)
        except OverflowError:
            return _JAXON_MISSING
    if isinstance(pytree, float):
        return np.double(pytree)
    if isinstance(pytree, complex):
        return np.cdouble(pytree)
    assert False


def _to_atom_non_reference_type(pytree: PyTree, downcast_to_base_types: tuple,
        py_to_np_types: set[JaxonPyNumeric]) -> _JaxonAtom | object:
    """Try to convert ``pytree`` to the internal representation if it is an object of a type
    that does not require references to be preserved. For example, python guarantees identity
    for all ``None`` objects; for other objects such as ``int`` or ``float`` references are
    not preserved by jaxon, as they cannot be relied upon anyway in python. Return
    ``JAXON_MISSING`` if ``pytree`` does not qualify as a non reference object. Note that
    for ``str`` this function returns ``JAXON_MISSING`` as it might be possible to save memory
    if jaxon attempts to preserve references to string objects."""
    if pytree is None:
        return _JaxonAtom(JAXON_NONE)
    if pytree is Ellipsis:
        return _JaxonAtom(JAXON_ELLIPSIS)
    py_numeric_type = _base_type(pytree, JAXON_PY_NUMERIC_TYPES, downcast_to_base_types)
    if py_numeric_type is not None:
        should_convert_to_numpy = py_numeric_type in py_to_np_types
        can_convert_to_numpy = _convert_py_numeric_to_numpy(pytree) is not _JAXON_MISSING
        if should_convert_to_numpy and can_convert_to_numpy:
            return _JAXON_MISSING  # act later in _to_atom_reference_type
        rep = repr(pytree)
        if py_numeric_type is complex:
            rep = rep[1:-1] if rep[0] == '(' else rep
        return _JaxonAtom(f"{py_numeric_type.__name__}({rep})")
    if isinstance(pytree, (range,)):  # range cannot be subclassed;
                                      # so downcast_to_base_types is irrelevant
        # remove unnecessary spaces
        return _JaxonAtom(repr(pytree).replace(" ", ""))
    return _JAXON_MISSING


def _to_atom_reference_type(pytree: PyTree,
        allow_dill: bool,
        dill_kwargs: dict,
        downcast_to_base_types: tuple[type, ...],
        py_to_np_types: set[JaxonPyNumeric],
        custom_marshalers: tuple[Marshaler, ...],
        parent_ids: frozenset[int],
        debug_path: str,
        cached_atoms: dict[int, _JaxonAtom]) -> _JaxonAtom:
    """Convert ``pytree`` recursively to the internal representation. Should only be called if
    ``_to_atom_non_reference_type`` returned ``JAXON_MISSING``."""
    # handle python types that are converted to numpy types
    py_numeric_type = _base_type(pytree, JAXON_PY_NUMERIC_TYPES, downcast_to_base_types)
    if py_numeric_type is not None:
        converted = _convert_py_numeric_to_numpy(pytree)
        assert not isinstance(converted, _JaxonMissing), "should have been handled before"
        return _JaxonAtom(converted)

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
        return _JaxonAtom("'" + pytree + "'")

    # handle arrays
    if _base_type(pytree, (_JAXON_JAX_ARRAY_TYPE,), downcast_to_base_types):
        dtype = pytree.dtype
        if dtype not in JAXON_NUMPY_NUMERIC_DTYPES:
            try:
                dtype = _JAXON_JAX_TO_NUMPY_TYPE[dtype.name]
            except KeyError:
                raise TypeError(f"unknown dtype {dtype.name!r}, of jax array at {debug_path!r}")
        return _JaxonAtom(np.asarray(pytree, dtype=dtype), JAXON_JAX_ARRAY,
                          typearg=pytree.dtype.name)
    if _base_type(pytree, (np.ndarray,), downcast_to_base_types):
        scalar_type = pytree.dtype.type
        if scalar_type not in JAXON_NUMPY_ARRAY_TYPES:
            raise TypeError(f"dtype {scalar_type!r} of numpy array at "
                            f"{debug_path!r} is not supported")
        return _JaxonAtom(pytree, JAXON_NUMPY_ARRAY)
    byte_buffer_type = _base_type(pytree, (bytes, bytearray), downcast_to_base_types)
    if byte_buffer_type is not None:
        return _JaxonAtom(np.void(pytree), byte_buffer_type.__name__)

    # handle containers and custom objects
    has_done_conversion = False
    typehint = ""
    container_type = _base_type(pytree, JAXON_PY_CONTAINER_TYPES, downcast_to_base_types)
    while container_type is None:  # try to convert to container type
        result = _marshal_custom_obj(pytree, custom_marshalers)
        if result is None:
            # conversion to container type failed
            break
        # the '#' indicates that the class uses the to_jaxon/from_jaxon
        # interface or another custom type conversion method
        typehint = "#" + result[0] + typehint
        pytree = result[1]
        has_done_conversion = True
        container_type = _base_type(pytree, JAXON_PY_CONTAINER_TYPES, downcast_to_base_types)
    if container_type is not None:
        typehint = container_type.__name__ + typehint
        debug_path += f"[{typehint}]"
        if isinstance(pytree, dict):
            data = _JaxonDict()
            for i, (dict_key, dict_value) in enumerate(pytree.items()):
                key_atom = _to_atom(dict_key, allow_dill, dill_kwargs, downcast_to_base_types,
                                    py_to_np_types, custom_marshalers, parent_ids,
                                    f"{debug_path}.key({i})", cached_atoms)
                dbgstr = f"{debug_path}.{_key_to_debugstring(dict_key, i)}"
                value_atom = _to_atom(dict_value, allow_dill, dill_kwargs, downcast_to_base_types,
                                      py_to_np_types, custom_marshalers, parent_ids, dbgstr,
                                      cached_atoms)
                data.data.append((key_atom, value_atom))
        else:
            data = _JaxonList()
            for i, item in enumerate(pytree):
                item_atom = _to_atom(item, allow_dill, dill_kwargs, downcast_to_base_types,
                                    py_to_np_types, custom_marshalers, parent_ids,
                                    f"{debug_path}({i})", cached_atoms)
                data.data.append(item_atom)
        return _JaxonAtom(data, typehint)
    if has_done_conversion:
        raise TypeError(f"Object at {debug_path!r} is not a valid jaxon container type; it was "
                         "returned by a custom type conversion, but is not an instance of dict, "
                         "list, tuple, set or frozenset or another object that can be converted.")

    # last resort: use dill for any other types if enabled
    # the '!' denotes that the object is serialized
    typehint = "!" + _get_qualified_name(pytree) + typehint
    debug_path += f"[{typehint}]"
    if allow_dill:
        return _JaxonAtom(np.array(dill.dumps(pytree, **dill_kwargs)), typehint)
    raise TypeError(f"Object at {debug_path!r} is not a valid jaxon type, but it can be "
                     "serialized if allow_dill is set to True.")


def _escape_attrib_path_ele(path: str) -> str:
    return path.replace("\\", "\\\\").replace("/", "\\/")


def _store_str_in_attrib(group, attrib_key_with_th: str, s: str) -> None:
    assert len(s) > 0, "string should not be of length 0 "
    encoded = s.encode('utf-8')
    dt = h5py.string_dtype(encoding='utf-8', length=len(encoded))
    group.attrs.create(attrib_key_with_th, data=np.bytes_(encoded), dtype=dt)


def _store_atom(group, atom: _JaxonAtom, attrib_key: str,
        stored_atoms: dict[int, str], group_path: str) -> None:
    """Recursively store the internal representation in the HDF5 file."""
    assert group_path[-1] == "/"
    target_group_path = stored_atoms.get(id(atom), _JAXON_MISSING)
    if target_group_path is not _JAXON_MISSING:
        _store_str_in_attrib(group, attrib_key, target_group_path)  # type: ignore
        return
    group_key = attrib_key  # if a sub group is generated
                            # it will have the same name as attrib_key
    sub_group_path = group_path + _escape_attrib_path_ele(group_key) + "/"
    if type(atom.data) is _JaxonDict:
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
                _store_atom(sub_group, key_atom, attrib_key_of_key, stored_atoms, sub_group_path)
            _store_atom(sub_group, value_atom, attrib_key_of_value, stored_atoms, sub_group_path)
        attrib_data = np.array([])
    elif type(atom.data) is _JaxonList:
        sub_group = group.create_group(group_key, track_order=True)
        for i, item_atom in enumerate(atom.data.data):
            _store_atom(sub_group, item_atom, str(i), stored_atoms, sub_group_path)
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
        if b'\0' in data and type(data) is not np.void:
            data = np.void(data)
            dtype = None
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
    attrib_key_with_th = attrib_key
    if atom.typehint is not None:
        attrib_key_with_th += f":{atom.typehint}"
        if atom.typearg is not None:
            attrib_key_with_th += f"[{atom.typearg}]"
    if type(attrib_data) is str:
        _store_str_in_attrib(group, attrib_key_with_th, attrib_data)
    else:
        group.attrs[attrib_key_with_th] = attrib_data
    group_path = group_path + _escape_attrib_path_ele(attrib_key_with_th)
    stored_atoms[id(atom)] = group_path


def save(path_or_file, pytree: PyTree,
         exact_python_numeric_types: bool = True,
         downcast_to_base_types: Iterable[type] | None = None,
         py_to_np_types: Iterable[JaxonPyNumeric] | None = None,
         allow_dill: bool = False,
         dill_kwargs: dict | None = None,
         custom_marshalers: tuple[Marshaler, ...] = tuple()) -> None:
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
    custom_marshalers = tuple(custom_marshalers)
    root_atom = _to_atom(pytree, allow_dill, dill_kwargs, downcast_to_base_types,
                         py_to_np_types, custom_marshalers, frozenset(), "", {})
    if hasattr(path_or_file, "seek") and hasattr(path_or_file, "truncate"):
        # when a file like object is provided
        # the file must be truncated like this because the "w"
        # mode does not seem to do this (bug?)
        path_or_file.seek(0)
        path_or_file.truncate()
    with h5py.File(path_or_file, 'w', track_order=True) as file:
        _store_atom(file, root_atom, JAXON_ROOT_GROUP_KEY, {}, "/")
