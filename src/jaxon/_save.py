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
import dataclasses
import numpy as np
import jax
import h5py
import dill

from ._common import (
    PyTree, Marshaler,
    JAXON_NP_NUMERIC_TYPES, JAXON_PY_NUMERIC_TYPES, JAXON_CONTAINER_TYPES,
    JAXON_JAX_ARRAY_TYPE, JAXON_NONE, JAXON_ELLIPSIS, JAXON_REF, JAXON_ROOT_GROUP_KEY,
    JAXON_DICT_KEY, JAXON_DICT_VALUE,
    JaxonAtom, JaxonDict, JaxonList, JaxonStorageHints,
    CircularPyTreeException,
    _JAXON_MISSING,
    _get_qualified_name, _key_to_debugstring,
)


def _base_type_name(obj, types, downcast_to_base_types):
    """Check if the type of `obj` is in `types` or if the user allowed downcasting to any
    of the types (if downcasting is possible)."""
    for t in types:
        if type(obj) is t or (type(obj) in downcast_to_base_types and isinstance(obj, t)):
            return t.__name__
    return None


def _encode_string(string):
    """All strings are stored as utf-8 fixed length strings."""
    encoded = string.encode("utf-8")
    return np.array(encoded, dtype=h5py.string_dtype("utf-8", len(encoded)))


def _marshal_dataclass(instance):
    return {field.name: getattr(instance, field.name) for field in dataclasses.fields(instance)}


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
    if dataclasses.is_dataclass(pytree):
        return _get_qualified_name(pytree), _marshal_dataclass(pytree)
    return None


def _to_atom(pytree: PyTree, allow_dill: bool, dill_kwargs: dict, downcast_to_base_types: tuple,
             py_to_np_types: tuple, custom_marshalers: tuple[Marshaler, ...],
             parent_ids: frozenset[int], debug_path: str,
             cached_atoms: dict[int, JaxonAtom]) -> JaxonAtom:
    """Recursively convert ``pytree`` to the internal representation. This function handles caching,
    which also enables correct reconstruction of references during loading. Also, this function
    prevents infinite recursion (by detecting circular references) and adds the original object
    id to the atom."""
    atom = _to_atom_non_reference_type(pytree, downcast_to_base_types, py_to_np_types)
    if atom is not _JAXON_MISSING:
        return JaxonAtom(atom.data, atom.typehint, id(pytree))  # type: ignore (cannot be of
                                                                # type JaxonMissing)
    # from here, the data items can be bigger and it is worthwhile to cache them
    result = cached_atoms.get(id(pytree), _JAXON_MISSING)
    if result is not _JAXON_MISSING:
        return result  # type: ignore (cannot be of type JaxonMissing)
    if id(pytree) in parent_ids:
        raise CircularPyTreeException(f"detected circular reference in pytree at {debug_path!r}")
    parent_ids = parent_ids | {id(pytree)}
    atom = _to_atom_reference_type(pytree, allow_dill, dill_kwargs, downcast_to_base_types,
        py_to_np_types, custom_marshalers, parent_ids, debug_path, cached_atoms)
    atom = JaxonAtom(atom.data, atom.typehint, id(pytree))
    cached_atoms[id(pytree)] = atom
    return atom


def _to_atom_non_reference_type(pytree: PyTree, downcast_to_base_types: tuple,
        py_to_np_types: tuple) -> JaxonAtom | object:
    """Try to convert ``pytree`` to the internal representation if it is an object of a type
    that does not require references to be preserved. For example, python guarantees identity
    for all ``None`` objects; for other objects such as ``int`` or ``np.int`` references are
    not preserved by jaxon, as they cannot be relied upon anyway in python. Return
    ``JAXON_MISSING`` if ``pytree`` does not qualify as a non reference object. Note that
    for ``str`` this function returns ``JAXON_MISSING`` as it might be possible to save memory
    if jaxon attempts to preserve references to string objects."""

    # handle simple scalar(-like) types
    if pytree is None:
        return JaxonAtom(JAXON_NONE)
    if pytree is ...:
        return JaxonAtom(JAXON_ELLIPSIS)
    np_numeric_type = _base_type_name(pytree, JAXON_NP_NUMERIC_TYPES, downcast_to_base_types)
    if np_numeric_type is not None:
        return JaxonAtom(pytree)
    py_numeric_type = _base_type_name(pytree, JAXON_PY_NUMERIC_TYPES, downcast_to_base_types)
    if py_numeric_type is not None:
        if isinstance(pytree, py_to_np_types):
            return JaxonAtom(pytree)
        rep = repr(pytree)
        if isinstance(pytree, complex):
            rep = rep[1:-1] if rep[0] == '(' else rep
        return JaxonAtom(f"{py_numeric_type}({rep})")
    if isinstance(pytree, (range, slice)):  # range, slice cannot be subclassed;
                                            # so downcast_to_base_types is irrelevant
        # remove unnecessary spaces which would cause parsing to fail
        return JaxonAtom(repr(pytree).replace(" ", ""))

    # can not be a small object
    return _JAXON_MISSING


def _to_atom_reference_type(pytree: PyTree, allow_dill: bool, dill_kwargs: dict,
        downcast_to_base_types: tuple, py_to_np_types: tuple, custom_marshalers: tuple[Marshaler, ...],
        parent_ids: frozenset[int], debug_path: str,
        cached_atoms: dict[int, JaxonAtom]) -> JaxonAtom:
    """Convert ``pytree`` recursively to the internal representation. Should only be called if
    ``_to_atom_non_reference_type`` returned ``JAXON_MISSING``."""
    # handle strings
    str_type = _base_type_name(pytree, (str,), downcast_to_base_types)
    if str_type is not None:
        # add quotation marks to avoid possible naming collision with type hint
        return JaxonAtom("'" + pytree + "'")

    # handle arrays
    if _base_type_name(pytree, (JAXON_JAX_ARRAY_TYPE,), downcast_to_base_types):
        return JaxonAtom(np.array(pytree), "jax.Array")
    if _base_type_name(pytree, (np.ndarray,), downcast_to_base_types):
        return JaxonAtom(pytree, "numpy.ndarray")
    byte_buffer_type = _base_type_name(pytree, (bytes, bytearray, memoryview),
                                       downcast_to_base_types)
    if byte_buffer_type is not None:
        return JaxonAtom(pytree if isinstance(pytree, memoryview) else memoryview(pytree),
                         byte_buffer_type)

    # handle containers and custom objects
    has_done_conversion = False
    typehint = ""
    container_type = _base_type_name(pytree, JAXON_CONTAINER_TYPES, downcast_to_base_types)
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
        container_type = _base_type_name(pytree, JAXON_CONTAINER_TYPES, downcast_to_base_types)
    if container_type is not None:
        typehint = container_type + typehint
        debug_path += f"[{typehint}]"
        if isinstance(pytree, dict):
            data = JaxonDict()
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
            data = JaxonList()
            for i, item in enumerate(pytree):
                item_atom = _to_atom(item, allow_dill, dill_kwargs, downcast_to_base_types,
                                    py_to_np_types, custom_marshalers, parent_ids,
                                    f"{debug_path}({i})", cached_atoms)
                data.data.append(item_atom)
        return JaxonAtom(data, typehint)
    if has_done_conversion:
        raise TypeError(f"Object at {debug_path!r} is not a valid jaxon container type; it was "
                         "returned by a custom type conversion, but is not an instance of dict, "
                         "list, tuple, set or frozenset or another object that can be converted.")

    # last resort: use dill for any other types if enabled
    # the '!' denotes that the object is serialized
    typehint = "!" + _get_qualified_name(pytree) + typehint
    debug_path += f"[{typehint}]"
    if allow_dill:
        return JaxonAtom(memoryview(dill.dumps(pytree, **dill_kwargs)), typehint)
    raise TypeError(f"Object at {debug_path!r} is not a valid jaxon type, but it can be "
                     "serialized if allow_dill is set to True.")


def _escape_attrib_path_ele(path: str) -> str:
    return path.replace("\\", "\\\\").replace("/", "\\/")


def _store_in_attrib(group, data: Any, group_key: str, atom: JaxonAtom,
                     stored_atoms: dict[int, str], group_path: str) -> None:
    if isinstance(data, str):
        group.attrs[group_key] = _encode_string(data)
    elif isinstance(data, (*JAXON_PY_NUMERIC_TYPES, *JAXON_NP_NUMERIC_TYPES, np.ndarray,
                           memoryview)):
        group.attrs[group_key] = data
    else:
        raise TypeError(f"unexpected internal jaxon data type {type(data)!r}")
    attrib_path = group_path + _escape_attrib_path_ele(group_key)
    stored_atoms[id(atom)] = attrib_path


def _store_atom(group, atom: JaxonAtom, group_key: str, storage_hints: dict,
                stored_atoms: dict[int, str], group_path: str) -> None:
    """Recursively store the internal representation in the HDF5 file."""
    assert group_path[-1] == "/"
    attrib_path = stored_atoms.get(id(atom), _JAXON_MISSING)
    if attrib_path is not _JAXON_MISSING:
        group.attrs[f"{group_key}:{JAXON_REF}"] = _encode_string(attrib_path)
    elif atom.typehint is None:
        _store_in_attrib(group, atom.data, group_key, atom, stored_atoms, group_path)
    elif isinstance(atom.data, JaxonDict):
        sub_group_path = group_path + _escape_attrib_path_ele(group_key) + "/"
        sub_group = group.create_group(group_key, track_order=True)
        for i, (key_atom, value_atom) in enumerate(atom.data.data):
            if key_atom.is_simple():
                group_key_of_value = key_atom.data
            else:
                # If the dict key atom is not simple it cannot be used directly
                # as the group key in the HDF5 file. So it must be stored
                # in another group attribute.
                group_key_of_value = f"{JAXON_DICT_VALUE}({i})"
                group_key_of_key = f"{JAXON_DICT_KEY}({i})"
                _store_atom(sub_group, key_atom, group_key_of_key, storage_hints, stored_atoms,
                            sub_group_path)
            _store_atom(sub_group, value_atom, group_key_of_value, storage_hints, stored_atoms,
                        sub_group_path)
        _store_in_attrib(group, atom.typehint, group_key, atom, stored_atoms, group_path)
    elif isinstance(atom.data, JaxonList):
        sub_group_path = group_path + _escape_attrib_path_ele(group_key) + "/"
        sub_group = group.create_group(group_key, track_order=True)
        for i, item_atom in enumerate(atom.data.data):
            _store_atom(sub_group, item_atom, str(i), storage_hints, stored_atoms, sub_group_path)
        _store_in_attrib(group, atom.typehint, group_key, atom, stored_atoms, group_path)
    elif isinstance(atom.data, (np.ndarray, memoryview)):
        storage_hint = storage_hints.get(atom.original_obj_id, None)
        if storage_hint is None or not storage_hint.store_in_dataset:
            # if it is desired to store the data in the attribute value
            # the typehint (which is always a string) must go into the group key
            _store_in_attrib(group, atom.data, f"{group_key}:{atom.typehint}", atom, stored_atoms,
                             group_path)
        else:
            _store_in_attrib(group, atom.typehint, group_key, atom, stored_atoms, group_path)
            group.create_dataset(group_key, data=atom.data)
    else:
        raise TypeError(f"unexpected internal jaxon data type {type(atom.data)!r}")


def save(path_or_file, pytree: PyTree,
         exact_python_numeric_types: bool = True,
         downcast_to_base_types: Iterable | None = None,
         py_to_np_types: Iterable | None = None,
         allow_dill: bool = False,
         dill_kwargs: dict | None = None,
         storage_hints: Iterable[tuple[Any, JaxonStorageHints]] | None = None,
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
    downcast_to_base_types : Iterable
        If a superclass of a supported base type is encountered in the pytree and is contained in
        this Iterable, it is converted to and stored as the supported base type. This means that
        it is also reconstructed as the supported base type when the file is loaded.
    py_to_np_types : Iterable
        Apply the behavior of ``exact_python_numeric_types`` only to the python types in the given
        Iterable. If not ``None``, ``exact_python_numeric_types`` will be ignored.
    allow_dill : bool, default=False
        Whether to allow ``dill`` for serializing unsupported objects.
    dill_kwargs : dict or None, optional
        Extra keyword arguments passed to ``dill.dumps`` if ``allow_dill`` is True.
    storage_hints : Iterable of tuple[Any, JaxonStorageHints], optional
        A list of hints for how to store numpy/jax arrays, bytes, bytearray and memoryview
        objects. The first member must be a reference to an object in `pytree` and the second
        specifies the corresponding ``JaxonStorageHints``. If the object is not found in
        the pytree, the hint is silently ignored.
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
            py_to_np_types = tuple()
        else:
            py_to_np_types = JAXON_PY_NUMERIC_TYPES
    else:
        py_to_np_types = tuple(py_to_np_types)
    if downcast_to_base_types is None:
        downcast_to_base_types = tuple()
    else:
        downcast_to_base_types = tuple(downcast_to_base_types)
    if storage_hints is None:
        storage_hints_converted = {}
    else:
        storage_hints_converted = {id(obj): hint for obj, hint in storage_hints}
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
        _store_atom(file, root_atom, JAXON_ROOT_GROUP_KEY, storage_hints_converted, {}, "/")
