# Copyright (C) 2025  Frank Hermann

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
Main module that provides the save and load functions.

Author
------
Frank Hermann
"""


from typing import Any, Iterable
from dataclasses import dataclass, field
import dataclasses
import importlib
import jax
import numpy as np
import h5py
import dill


# note that the following lists of types do not represent what is supported by jaxon
# (refer to the README)
JAXON_NP_NUMERIC_TYPE_NAMES = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "float16", "float32", "float64", "float128", "complex64", "complex128",
    "bool")  # supported python dtypes
JAXON_NP_NUMERIC_TYPES = tuple(getattr(np, typename) for typename in JAXON_NP_NUMERIC_TYPE_NAMES
                               if hasattr(np, typename))
JAXON_PY_NUMERIC_TYPES = (int, float, bool, complex)  # supported python numeric types
JAXON_CONTAINER_TYPES = (list, tuple, dict, set, frozenset)  # supported python container types

# get the type of a jax array (in a version-independent way)
# it is used to detect jax arrays
JAXON_JAX_ARRAY_TYPE = type(jax.numpy.array([]))


# the following are keywords which are used in the hd5f file
JAXON_NONE = "None"  # used to encode python `None`
JAXON_ELLIPSIS = "Ellipsis"  # used to encode python `...`
JAXON_DICT_KEY = "key"  # used to indicate that this hd5f attribute stores
                        # the key of another attribute in the same group
                        # (only used if necessary)
JAXON_DICT_VALUE = "value" # used to indicate that this hd5f attribute stores
                           # a dict value (only used iff `JAXON_DICT_KEY` is used)
JAXON_ROOT_GROUP_KEY = "JAXON_ROOT"  # hd5f root group name (might be followed by typehint of
                                     # the root object)


class CircularPytreeException(Exception):
    """Raised when a circular reference (reference to a parent object) was detected."""


@dataclass
class JaxonDict:
    """Internal representation of a dict."""
    data: list[tuple['JaxonAtom', 'JaxonAtom']] = field(default_factory=list)


@dataclass
class JaxonList:
    """Internal representation of a list."""
    data: list['JaxonAtom'] = field(default_factory=list)


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
        simple atoms can be used as group or attribute keys in the hd5f file."""
        return self.typehint is None and type(self.data) is str and "\0" not in self.data


@dataclass
class JaxonStorageHints:
    """If the field `store_in_dataset` is `True` the associated data will be stored in an hd5f
    dataset. Otherwise, it will be stored in an hd5f attribute."""
    store_in_dataset: bool


def _get_qualified_name(obj):
    """The returned name fully identifies the class of the object so that a new object can be
    instantiated later during loading (see `_create_instance`)."""
    return type(obj).__module__ + "." + type(obj).__qualname__


def _create_instance(qualified_name: str):
    """Create a new instance of the class identified by `qualified_name` that was returned
    by `_get_qualified_name`."""
    parts = qualified_name.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls.__new__(cls)


def _range_from_repr(repr_content):
    fields = repr_content.split(",")
    assert len(fields) in (2, 3)
    return range(*[int(f) for f in fields])


def _slice_from_repr(repr_content):
    fields = repr_content.split(",")
    assert len(fields) == 3
    return slice(*[(None if f == "None" else int(f)) for f in fields])


def _base_type_name(obj, types, downcast_to_base_types):
    """Check if the type of `obj` is in `types` or if the user allowed downcasting to any
    of the types (if downcasting is possible)."""
    for t in types:
        if type(obj) is t or (type(obj) in downcast_to_base_types and isinstance(obj, t)):
            return t.__name__
    return None


def _encode_string(string):
    """All string are stored as utf-8 fixed length strings."""
    encoded = string.encode("utf-8")
    return np.array(encoded, dtype=h5py.string_dtype("utf-8", len(encoded)))


def _decode_string(buffer):
    """All string are stored as utf-8 fixed length strings."""
    return buffer.decode("utf-8")


def _dataclass_to_container(instance):
    return {field.name: getattr(instance, field.name) for field in dataclasses.fields(instance)}


def _custom_obj_to_container(pytree):
    """This function is called to handle custom types. The first member in the returned
    tuple indicates if conversion took place, the second gives the result of the conversion."""
    if hasattr(pytree, "to_jaxon"):
        return True, pytree.to_jaxon()
    if dataclasses.is_dataclass(pytree):
        return True, _dataclass_to_container(pytree)
    return False, None


def _container_to_dataclass(container, instance):
    assert type(container) is dict, "expected dict container for dataclass"
    for field_name, field_value in container.items():
        object.__setattr__(instance, field_name, field_value)


def _container_to_custom_obj(container, instance):
    """Opposite of `_custom_obj_to_container`. Returns True if conversion took place."""
    if hasattr(instance, "from_jaxon"):
        instance.from_jaxon(container)
    elif dataclasses.is_dataclass(instance):
        _container_to_dataclass(container, instance)
    else:
        return False
    return True


def to_atom(pytree, allow_dill=False, dill_kwargs=None, downcast_to_base_types: tuple = tuple(),
             py_to_np_types: tuple = tuple(), parent_objects=None, debug_path="") -> JaxonAtom:
    """Recursively convert `pytree` to the internal representation. This is the entry point for
    `_to_atom` which does the actual work. Here, only the `id` of `pytree` is added to the
    returned `atom`."""
    atom = _to_atom(pytree, allow_dill, dill_kwargs, downcast_to_base_types, py_to_np_types,
                    parent_objects, debug_path)
    return JaxonAtom(atom.data, atom.typehint, id(pytree))


def _key_to_debugstring(dict_key, i):
    if isinstance(dict_key, (str, int, float, bool, complex)):
        return repr(dict_key)
    return f"{(i)}"


def _to_atom(pytree, allow_dill, dill_kwargs, downcast_to_base_types, py_to_np_types,
             parent_objects, debug_path) -> JaxonAtom:
    """Recursively convert `pytree` to the internal representation."""

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
            rep = rep[1:-1]  # remove unnecessary brackets
        return JaxonAtom(f"{py_numeric_type}({rep})")
    other_repr_type = _base_type_name(pytree, (range, slice), downcast_to_base_types)
    if other_repr_type is not None:
        typehint = repr(pytree)
        if isinstance(pytree, (range, slice)):
            # remove unnecessary spaces which would cause parsing to fail
            typehint = typehint.replace(" ", "")
        return JaxonAtom(typehint)
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

    # handle container types first
    if parent_objects is None:
        parent_objects = [pytree]  # root node
    elif any(pytree is p for p in parent_objects):
        raise CircularPytreeException(f"detected circular reference in pytree at {debug_path!r}")
    else:
        parent_objects = parent_objects + [pytree]  # Descend. Need new list of parents here.
    is_custom_type = False
    typehint = ""
    container_type = _base_type_name(pytree, JAXON_CONTAINER_TYPES, downcast_to_base_types)
    while container_type is None:
        # the '#' indicates that the class uses the to_jaxon/from_jaxon interface
        # or another custom type conversion method
        new_typehint = "#" + _get_qualified_name(pytree) + typehint
        success, new_pytree = _custom_obj_to_container(pytree)
        if not success:
            break
        typehint = new_typehint
        pytree = new_pytree
        is_custom_type = True
        parent_objects += [pytree]
        container_type = _base_type_name(pytree, JAXON_CONTAINER_TYPES, downcast_to_base_types)
    if container_type is not None:
        typehint = container_type + typehint
        debug_path += f"[{typehint}]"
        if isinstance(pytree, dict):
            data = JaxonDict()
            for i, (dict_key, dict_value) in enumerate(pytree.items()):
                key_atom = to_atom(dict_key, allow_dill, dill_kwargs, downcast_to_base_types,
                                   py_to_np_types, parent_objects, f"{debug_path}.key({i})")
                dbgstr = f"{debug_path}.{_key_to_debugstring(dict_key, i)}"
                value_atom = to_atom(dict_value, allow_dill, dill_kwargs, downcast_to_base_types,
                                     py_to_np_types, parent_objects, dbgstr)
                data.data.append((key_atom, value_atom))
        else:
            data = JaxonList()
            for i, item in enumerate(pytree):
                item_atom = to_atom(item, allow_dill, dill_kwargs, downcast_to_base_types,
                                    py_to_np_types, parent_objects, f"{debug_path}({i})")
                data.data.append(item_atom)
        return JaxonAtom(data, typehint)
    if is_custom_type:
        raise TypeError(f"Object at {debug_path!r} is not a valid jaxon container type; it was "
                         "returned by a custom type conversion, but is not an instance of dict, "
                         "list, tuple, set or frozenset or another object that can be converted.")

    # last resort: use dill for any other types if enabled
    # the '!' denotes that the object is serialized
    typehint = "!" + _get_qualified_name(pytree) + typehint
    debug_path += f"[{typehint}]"
    if allow_dill:
        if dill_kwargs is None:
            dill_kwargs = {}
        return JaxonAtom(memoryview(dill.dumps(pytree, **dill_kwargs)), typehint)
    raise TypeError(f"Object at {debug_path!r} is not a valid jaxon type, but it can be "
                     "serialized if allow_dill is set to True.")


def _store_in_attrib(group, data, group_key):
    if isinstance(data, str):
        group.attrs[group_key] = _encode_string(data)
    elif isinstance(data, (*JAXON_PY_NUMERIC_TYPES, *JAXON_NP_NUMERIC_TYPES, np.ndarray,
                           memoryview)):
        group.attrs[group_key] = data
    else:
        assert False, f"unexpected internal jaxon data type {type(data)!r}"


def _store_atom(group, atom, group_key, storage_hints):
    """Recursively store the internal representation in the hd5f file."""
    if atom.typehint is None:
        _store_in_attrib(group, atom.data, group_key)
    elif isinstance(atom.data, JaxonDict):
        sub_group = group.create_group(group_key, track_order=True)
        for i, (key_atom, value_atom) in enumerate(atom.data.data):
            if key_atom.is_simple():
                group_key_of_value = key_atom.data
            else:
                # If the dict key atom is not simple it cannot be used directly
                # as the group key in the hd5f file. So it must be stored
                # in another group attribute.
                group_key_of_value = f"{JAXON_DICT_VALUE}({i})"
                group_key_of_key = f"{JAXON_DICT_KEY}({i})"
                _store_atom(sub_group, key_atom, group_key_of_key, storage_hints)
            _store_atom(sub_group, value_atom, group_key_of_value, storage_hints)
        _store_in_attrib(group, atom.typehint, group_key)
    elif isinstance(atom.data, JaxonList):
        sub_group = group.create_group(group_key, track_order=True)
        for i, item_atom in enumerate(atom.data.data):
            _store_atom(sub_group, item_atom, str(i), storage_hints)
        _store_in_attrib(group, atom.typehint, group_key)
    elif isinstance(atom.data, (np.ndarray, memoryview)):
        storage_hint = storage_hints.get(atom.original_obj_id, None)
        if storage_hint is None or not storage_hint.store_in_dataset:
            # if it is desired to store the data in the attribute value
            # the typehint (which is always a string) must go into the group key
            _store_in_attrib(group, atom.data, f"{group_key}:{atom.typehint}")
        else:
            _store_in_attrib(group, atom.typehint, group_key)
            group.create_dataset(group_key, data=atom.data)
    else:
        assert False, f"unexpected internal jaxon data type {type(atom.data)!r}"


def _simple_atom_from_data_str(typehint_or_data: str):
    """Tries to interpret `typehint_or_data` as the `data` part of a simple
    atom (minus the restriction that the atom cannot contain null chars).
    The `typehint_or_data` comes from an attribute value or key. Return a tuple
    where the first member indicates if this interpretation is possible and the
    second is the data if yes."""
    if typehint_or_data == JAXON_NONE:
        return True, None
    if typehint_or_data == JAXON_ELLIPSIS:
        return True, ...
    if typehint_or_data[0] == "'":
        assert len(typehint_or_data) >= 2 and typehint_or_data[-1] == "'", \
               "string parsing error: unexpected termination"
        return True, typehint_or_data[1:-1]
    other_repr_types = [(int, None), (float, None), (bool, None), (complex, None),
                        (range, _range_from_repr), (slice, _slice_from_repr)]
    for primitive, parser in other_repr_types:
        # here, we parse primitives that were saved with exact_python_types=True
        type_name = primitive.__name__
        if not typehint_or_data.startswith(type_name):
            continue
        assert typehint_or_data[len(type_name)] == "(" and typehint_or_data[-1] == ")", \
               "primitive parsing error"
        repr_content = typehint_or_data[len(type_name) + 1:-1]
        if parser is not None:
            return True, parser(repr_content)
        return True, primitive(repr_content)
    return False, None


def _get_group_key_and_typehint(group_key_with_typehint):
    """Separates the actual key from a possibly added typehint."""
    # attention must be paid here as the colons (which separate the typehint)
    # are not escaped in strings
    if group_key_with_typehint[-1] == "'":
        assert group_key_with_typehint[0] == "'", "string format error"
        # single string without typehint
        return group_key_with_typehint, None
    for i in reversed(range(len(group_key_with_typehint))):
        ch = group_key_with_typehint[i]
        if ch == ":":
            group_key = group_key_with_typehint[:i]
            th = group_key_with_typehint[i + 1:]
            return group_key, th
    # something else (like int(42)) which is not a string and is used as group key
    return group_key_with_typehint, None


def _load_data(group, attr_value, group_key_with_th, has_key_th):
    if has_key_th:
        # presence of a type hint in the key implies that the data resides
        # in the attribute value (load it it if it's not already loaded)
        if attr_value is None:
            return group.attrs[group_key_with_th]
        return attr_value
    # otherwise, it resides in a dataset
    return group[group_key_with_th][()]


def _load(group, group_key_and_th, allow_dill=False, dill_kwargs=None, debug_path=""):
    """Recursively load the pytree from the hd5f file. Here, `group` is an h5py group object,
    the `group_key_and_th` is the group key (including a possible type hint) which must be
    a valid key in the group's attribute dict."""
    _, th = _get_group_key_and_typehint(group_key_and_th)
    has_key_th = th is not None
    attr_value = None  # if None, it will be loaded later on demand (if necessary)
    if not has_key_th:
        attr_value = group.attrs[group_key_and_th]
        if type(attr_value) in JAXON_NP_NUMERIC_TYPES:
            # here we also load primitives like int or float if they were saved
            # with exact_python_types=False
            return attr_value
        # if the typehint (th) is not specified in the group_key
        # and the attribute is not one of JAXON_NP_NUMERIC_TYPES
        # the attr_value either encodes the typehint or data
        attr_dtype = group.attrs.get_id(group_key_and_th).dtype
        string_dtype = h5py.check_string_dtype(attr_dtype)
        assert string_dtype is not None, "unexpected hdf5 attribute type"
        assert string_dtype.length is not None, "expected a fixed length string"
        assert string_dtype.encoding == "utf-8", "unexpected string encoding"
        th_or_data = _decode_string(attr_value)
        is_simple_atom, pytree = _simple_atom_from_data_str(th_or_data)
        if is_simple_atom:
            return pytree
        # if it's not a simple atom, it must be a typehint
        th = th_or_data

    # handle arrays
    if th == "bytes":
        return bytes(_load_data(group, attr_value, group_key_and_th, has_key_th))
    if th == "bytearray":
        return bytearray(_load_data(group, attr_value, group_key_and_th, has_key_th))
    if th == "memoryview":
        return memoryview(_load_data(group, attr_value, group_key_and_th, has_key_th))
    if th == "numpy.ndarray":
        return _load_data(group, attr_value, group_key_and_th, has_key_th)
    if th == "jax.Array":
        return jax.numpy.array(_load_data(group, attr_value, group_key_and_th, has_key_th))

    # handle serialized types
    if th[0] == "!":
        if not allow_dill:
            raise ValueError(f"cannot load serialized object at {debug_path!r}, "
                              "as allow_dill=False")
        if dill_kwargs is None:
            dill_kwargs = {}
        data = _load_data(group, attr_value, group_key_and_th, has_key_th)
        return dill.loads(data, **dill_kwargs)

    # handle container types
    debug_path = f"{debug_path}[{th}]"
    types = th.split("#")
    if types[0] == "dict":
        sub_group = group[group_key_and_th]
        pytree = {}
        for i, sub_group_key in enumerate(sub_group.attrs):
            if sub_group_key.startswith("value"):
                continue  # loaded later when if corresponding key is read
            if sub_group_key.startswith("key"):
                assert sub_group_key[len("key")] == "(" and sub_group_key[-1] == ")"
                group_key_of_value = f"value({int(sub_group_key[len('key')+1:-1])})"
                dbgstr = f"{debug_path}.key({i})"
                dict_key = _load(sub_group, sub_group_key, allow_dill, dill_kwargs, dbgstr)
            else:
                # assume that the key is a simple atom (fully represented by sub_group_key)
                sub_group_key_data, _ = _get_group_key_and_typehint(sub_group_key)
                is_simple_atom, dict_key = _simple_atom_from_data_str(sub_group_key_data)
                assert is_simple_atom, f"expected simple atom for sub group key {sub_group_key!r}"
                group_key_of_value = sub_group_key
            dbgstr = f"{debug_path}.{_key_to_debugstring(dict_key, i)}"
            pytree[dict_key] = _load(sub_group, group_key_of_value, allow_dill,
                                     dill_kwargs, dbgstr)
    elif types[0] in ("list", "tuple", "set", "frozenset"):
        sub_group = group[group_key_and_th]
        pytree = [_load(sub_group, sub_group_key, allow_dill, dill_kwargs, f"{debug_path}({i})")
                  for i, sub_group_key in enumerate(sub_group.attrs)]
        if types[0] == "tuple":
            pytree = tuple(pytree)
        if types[0] == "set":
            pytree = set(pytree)
        if types[0] == "frozenset":
            pytree = frozenset(pytree)
    else:
        raise ValueError(f"type of object at {debug_path!r} not understood")
    for qualified_name in types[1:]:
        instance = _create_instance(qualified_name)
        success = _container_to_custom_obj(pytree, instance)
        pytree = instance
        if not success:
            raise ValueError(f"cannot load object at {debug_path!r}, as type "
                             f"{_get_qualified_name!r} has not attribute from_jaxon")
    return pytree


def save(path, pytree,
         exact_python_numeric_types: bool = True,
         downcast_to_base_types: Iterable | None = None,
         py_to_np_types: Iterable | None = None,
         allow_dill: bool = False,
         dill_kwargs: dict | None = None,
         storage_hints: Iterable[tuple[Any, JaxonStorageHints]] | None = None):
    """
    Save a pytree in a human readable format in an hd5f file with the specified path.
    If the file already exists, it's contents are overwritten.

    Parameters
    ----------
    path :
        The file path where the pytree will be saved.
    pytree :
        The pytree object to be saved. Can contain nested structures of arrays, lists,
        dicts, etc. (see README)
    exact_python_numeric_types : bool, default=True
        If False, the types int, float, bool and complex will be converted implicitly to
        np.int64, np.float64, np.bool and np.complex128 respectively and stored as the
        corresponding hd5f binary type. If the file is loaded, the types will be the numpy
        (not python) types. If True, they are stored as strings and fully reconstructed
        when loading.
    downcast_to_base_types : Iterable
        If a superclass of a supported base type is encountered in the pytree and is contained in
        this Iterable, it is converted to and stored as the supported base type. This means that
        it is also reconstructed as the supported base type when the file is loaded.
    py_to_np_types : Iterable
        Apply the behavior of `exact_python_numeric_types` only to the python types in the given
        Iterable. If not `None`, `exact_python_numeric_types` will be ignored.
    allow_dill : bool, default=False
        Whether to allow `dill` for serializing unsupported objects.
    dill_kwargs : dict or None, optional
        Extra keyword arguments passed to `dill.dumps` if `allow_dill` is True.
    storage_hints : Iterable of tuple[Any, JaxonStorageHints], optional
        A list of hints for how to store numpy/jax arrays, bytes, bytearray and memoryview
        objects. The first member must be a reference to an object in `pytree` and the second
        specifies the corresponding `JaxonStorageHints`. If the object is not found in
        the pytree, the hint is silently ignored.

    Returns
    -------
    None
        This function does not return anything. It writes data to the specified path.

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
    root_atom = to_atom(pytree, allow_dill, dill_kwargs, downcast_to_base_types, py_to_np_types)
    with h5py.File(path, 'w', track_order=True) as file:
        _store_atom(file, root_atom, JAXON_ROOT_GROUP_KEY, storage_hints_converted)


def load(path, allow_dill: bool = False, dill_kwargs: dict | None = None):
    """
    Load a pytree from an hd5f file.

    Parameters
    ----------
    path : str or Path
        The file path from which to load the pytree.
    allow_dill : bool, default=False
        Whether to allow loading objects serialized with `dill`. If a serialized object is
        encountered and this argument is `False`, an error is raised. 
    dill_kwargs : dict or None, optional
        Extra keyword arguments passed to `dill.loads` if `allow_dill` is True.

    Notes
    -----
    - This function expects the file format produced by the `save` function.
    """
    with h5py.File(path, 'r') as file:
        # a type hint might have been added to the JAXON_ROOT_GROUP_KEY
        group_key = next((group_key for group_key in file.attrs
                          if group_key.startswith(JAXON_ROOT_GROUP_KEY)), None)
        assert group_key is not None, "jaxon root group not found"
        return _load(file, group_key, allow_dill=allow_dill, dill_kwargs=dill_kwargs)
