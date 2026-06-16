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
Main module that provides the save and load functions.

Author
------
Frank Hermann
"""


from typing import Any, Iterable, Callable
from dataclasses import dataclass
import dataclasses
import importlib
import warnings
import jax
import numpy as np
import h5py
import dill


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


class JaxonNotLoaded:
    """Placeholder object used to indicate an object that has not been loaded, either
    on user request or because it is missing in the hdf5 file."""

    def __repr__(self):
        return "JAXON_NOT_LOADED"


class _DictKeyPathElement:
    """Flag object path element to indicate that loader descended into a dict key."""

    def __repr__(self):
        return "_DICT_KEY_PATH_ELEMENT"


JAXON_NOT_LOADED = JaxonNotLoaded()
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
        simple atoms can be used as group or attribute keys in the hd5f file."""
        return self.typehint is None and type(self.data) is str and "\0" not in self.data


@dataclass
class JaxonStorageHints:
    """If the field `store_in_dataset` is `True` the associated data will be stored in an hd5f
    dataset. Otherwise, it will be stored in an hd5f attribute."""
    store_in_dataset: bool


def has_common_prefix(path: Iterable, other_path: Iterable) -> bool:
    """Checks if the two Iterables start with the same values. If one of the Iterables
    is longer then the additional items are ignored."""
    return all(map(lambda ab: ab[0] == ab[1], zip(path, other_path)))


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
    if len(fields) not in (2, 3):
        raise ValueError(f"cannot parse range representation: {repr_content!r}")
    return range(*[int(f) for f in fields])


def _slice_from_repr(repr_content):
    fields = repr_content.split(",")
    if len(fields) != 3:
        raise ValueError(f"cannot parse slice representation: {repr_content!r}")
    return slice(*[(None if f == "None" else int(f)) for f in fields])


def _bool_from_repr(repr_content):
    if repr_content == "True":
        return True
    if repr_content == "False":
        return False
    raise ValueError(f"unexpected boolean string representation: {repr_content!r}")


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


def _unmarshal_dataclass(container: PyTree, instance: PyTree, allow_missing_fields: bool,
        allow_unknown_fields: bool) -> PyTree:
    if type(container) is not dict:
        raise JaxonError("expected dict container for dataclass")
    fields = dataclasses.fields(instance)
    field_names = {f.name for f in fields}
    available_field_names = container.keys()
    missing_fields = available_field_names - field_names
    if missing_fields:
        message = f"The following fields in {_get_qualified_name(instance)!r} are present in " \
                   "the hdf5 file but are missing in the class " \
                  f"definition: {", ".join(missing_fields)}"
        if allow_missing_fields:
            warnings.warn(message, JaxonFormatWarning)
        else:
            raise ValueError(message + ". To omit this error run with allow_missing_fields=True.")
    unknown_fields = field_names - available_field_names
    if unknown_fields:
        message = f"the following fields in {_get_qualified_name(instance)!r} are present in " \
                   "the class definition but are missing in the hdf5 " \
                  f"file: {", ".join(unknown_fields)}"
        if allow_unknown_fields:
            warnings.warn(message, JaxonFormatWarning)
        else:
            raise ValueError(message + "\n. To omit this error run with "
                "allow_unknown_fields=True. Missing fields will be initialized using the "
                "default_factory or default value. If both are missing JaxonNotLoaded will be "
                "used as a placeholder. The __post_init__() logic is never triggered.")
    for field in fields:
        try:
            val = container[field.name]
        except KeyError:
            if field.default_factory is not dataclasses.MISSING:
                val = field.default_factory()
            elif field.default is not dataclasses.MISSING:
                val = field.default
            else:
                val = JAXON_NOT_LOADED
        # use object.__setattr__ to make it work even if the dataclass is frozen
        object.__setattr__(instance, field.name, val)


def _unmarshal_custom_obj(qualified_name: str, container: PyTree,
        custom_unmarshalers: tuple[Unmarshaler, ...], allow_missing_fields: bool,
        allow_unknown_fields: bool) -> PyTree:
    """Opposite of ``_marshal_custom_obj``. Returns the unmarshaled custom object
    if successful or ``None`` otherwise."""
    for custom_unmarshaler in custom_unmarshalers:
        result = custom_unmarshaler(qualified_name, container)
        if result is not None:
            return result
    try:
        instance = _create_instance(qualified_name)
    except ValueError as e:
        raise ValueError(f"Failed to instantiate class {qualified_name!r}. This "
                          "problem could be solved by providing a custom unmarshaler.") from e
    if hasattr(instance, "from_jaxon"):
        instance.from_jaxon(container)
    elif dataclasses.is_dataclass(instance):
        _unmarshal_dataclass(container, instance, allow_missing_fields, allow_unknown_fields)
    else:
        return None
    return instance


def _to_atom(pytree: PyTree, allow_dill: bool, dill_kwargs: dict, downcast_to_base_types: tuple,
             py_to_np_types: tuple, custom_marshalers: tuple[Marshaler, ...],
             parent_objects: tuple[PyTree, ...], debug_path: str,
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
    if any(pytree is p for p in parent_objects):
        raise CircularPyTreeException(f"detected circular reference in pytree at {debug_path!r}")
    parent_objects = (*parent_objects, pytree)
    atom = _to_atom_reference_type(pytree, allow_dill, dill_kwargs, downcast_to_base_types,
        py_to_np_types, custom_marshalers, parent_objects, debug_path, cached_atoms)
    atom = JaxonAtom(atom.data, atom.typehint, id(pytree))
    cached_atoms[id(pytree)] = atom
    return atom


def _key_to_debugstring(dict_key, i) -> str:
    if isinstance(dict_key, (str, int, float, bool, complex)):
        return repr(dict_key)
    return f"{(i)}"


def _to_atom_non_reference_type(pytree: PyTree, downcast_to_base_types: tuple,
        py_to_np_types: tuple) -> JaxonAtom | _JaxonMissing:
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
            rep = rep[1:-1]  # remove unnecessary brackets
        return JaxonAtom(f"{py_numeric_type}({rep})")
    if isinstance(pytree, (range, slice)):  # range, slice cannot be subclassed;
                                            # so downcast_to_base_types is irrelevant
        # remove unnecessary spaces which would cause parsing to fail
        return JaxonAtom(repr(pytree).replace(" ", ""))

    # can not be a small object
    return _JAXON_MISSING


def _to_atom_reference_type(pytree: PyTree, allow_dill: bool, dill_kwargs: dict,
        downcast_to_base_types: tuple, py_to_np_types: tuple, custom_marshalers: tuple[Marshaler, ...],
        parent_objects: tuple[PyTree, ...], debug_path: str,
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
                                   py_to_np_types, custom_marshalers, parent_objects,
                                   f"{debug_path}.key({i})", cached_atoms)
                dbgstr = f"{debug_path}.{_key_to_debugstring(dict_key, i)}"
                value_atom = _to_atom(dict_value, allow_dill, dill_kwargs, downcast_to_base_types,
                                     py_to_np_types, custom_marshalers, parent_objects, dbgstr,
                                     cached_atoms)
                data.data.append((key_atom, value_atom))
        else:
            data = JaxonList()
            for i, item in enumerate(pytree):
                item_atom = _to_atom(item, allow_dill, dill_kwargs, downcast_to_base_types,
                                    py_to_np_types, custom_marshalers, parent_objects,
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
        assert False, f"unexpected internal jaxon data type {type(data)!r}"
    attrib_path = group_path + _escape_attrib_path_ele(group_key)
    stored_atoms[id(atom)] = attrib_path


def _store_atom(group, atom, group_key, storage_hints, stored_atoms: dict[int, str],
                group_path: str):
    """Recursively store the internal representation in the hd5f file."""
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
                # as the group key in the hd5f file. So it must be stored
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
        if len(typehint_or_data) < 2 or typehint_or_data[-1] != "'":
            raise ValueError(f"cannot parse string: unexpected termination in {typehint_or_data!r}")
        return True, typehint_or_data[1:-1]
    other_repr_types = [(int, None), (float, None), (bool, _bool_from_repr), (complex, None),
                        (range, _range_from_repr), (slice, _slice_from_repr)]
    for primitive, parser in other_repr_types:
        # here, we parse primitives that were saved with exact_python_types=True
        type_name = primitive.__name__
        if not typehint_or_data.startswith(type_name):
            continue
        if typehint_or_data[len(type_name)] != "(" or typehint_or_data[-1] != ")":
            raise ValueError(f"cannot parse {type_name} representation: {typehint_or_data!r}")
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
        if group_key_with_typehint[0] != "'":
            raise JaxonError("string format error")
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
        # in the attribute value (load it if it is not already loaded)
        if attr_value is None:
            return group.attrs[group_key_with_th]
        return attr_value
    # otherwise, it resides in a dataset
    return group[group_key_with_th][()]


def _parse_key_or_val(group_key: str, prefix: str) -> int:
    if group_key[len(prefix)] != "(":
        raise JaxonError(f"malformed group key {group_key!r}")
    return int(group_key[len(prefix) + 1:group_key.find(")")])


def _load(group, group_key_and_th: str, allow_dill: bool, dill_kwargs: dict,
          debug_path: str, custom_unmarshalers: tuple[Unmarshaler, ...],
          allow_missing_fields: bool, allow_unknown_fields: bool,
          load_filter: LoadFilter, parents: list[_PathElement], hdf5_path: tuple[str, ...],
          loaded_objects: dict[tuple[str, ...], PyTree],
          currently_loading_object: set[tuple[str, ...]]) -> PyTree:
    """Wrapper for _do_load(...) that returns the reference if it is already loaded by using a
    cache. Uses the ``hdf5_path`` as cache key. Raises an error if there is a circular reference
    during loading."""
    hdf5_path = (*hdf5_path, group_key_and_th)
    result = loaded_objects.get(hdf5_path, _JAXON_MISSING)
    if result is not _JAXON_MISSING:
        # available in cache
        return result
    if hdf5_path in currently_loading_object:
        raise CircularPyTreeException()
    currently_loading_object.add(hdf5_path)
    result = _do_load(group, group_key_and_th, allow_dill, dill_kwargs,
        debug_path, custom_unmarshalers, allow_missing_fields, allow_unknown_fields,
        load_filter, parents, hdf5_path, loaded_objects, currently_loading_object)
    currently_loading_object.remove(hdf5_path)
    if isinstance(result, _JaxonLoadedFromReferenceWrapper):
        # indicates that the result was loaded from a reference
        # since references to references are not allowed in jaxon
        # it is not necessary to add the path to loaded_objects
        result = result.pytree
    else:
        loaded_objects[hdf5_path] = result
    return result


def _tokenize_attrib_path(path: str) -> tuple[str]:
    """Return a list of the path elements of the reference path. This essentially splits the
    string at `/` while ignoring escaped `/` characters and unescaping the returned path
    elements. This function also checks that the path starts with a `/`."""
    if path[0] != "/":
        raise JaxonError("attribute path must start with a /")
    buf = ""
    res = []
    escape_next_char = False
    for c in path[1:]:
        if not escape_next_char:
            if c == "\\":
                escape_next_char = True
                continue
            if c == "/":
                res.append(buf)
                buf = ""
                continue
        escape_next_char = False
        buf += c
    res.append(buf)
    return tuple(res)


def _do_load(group, group_key_and_th: str, allow_dill: bool, dill_kwargs: dict,
             debug_path: str, custom_unmarshalers: tuple[Unmarshaler, ...],
             allow_missing_fields: bool, allow_unknown_fields: bool,
             load_filter: LoadFilter, parents: list[_PathElement], hdf5_path: tuple[str, ...],
             loaded_objects: dict[tuple[str, ...], PyTree],
             currently_loading_object: set[tuple[str, ...]]) -> PyTree:
    """Recursively load the pytree from the hd5f file. Here, `group` is an h5py group object,
    the `group_key_and_th` is the group key (including a possible type hint) which must be
    a valid key in the group's attribute dict."""
    if not any(p is _DICT_KEY_PATH_ELEMENT for p in parents) and not load_filter(parents):
        return JAXON_NOT_LOADED
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
        if string_dtype is None:
            raise JaxonError("unexpected hdf5 attribute type")
        if string_dtype.length is None:
            raise JaxonError("expected a fixed length string")
        if string_dtype.encoding != "utf-8":
            raise JaxonError("unexpected string encoding")
        th_or_data = _decode_string(attr_value)
        is_simple_atom, pytree = _simple_atom_from_data_str(th_or_data)
        if is_simple_atom:
            return pytree
        # if it's not a simple atom, it must be a typehint
        th = th_or_data

    # handle arrays
    if th == JAXON_REF:
        # attribute cannot be loaded already as typehint is always
        # in the keys for references
        attrib_path = _decode_string(group.attrs[group_key_and_th])
        attrib_path_eles = _tokenize_attrib_path(attrib_path)
        target_group = group.file
        for path_element in attrib_path_eles[:-1]:
            target_group = target_group[path_element]
        res = _load(target_group, attrib_path_eles[-1], allow_dill, dill_kwargs, debug_path,
            custom_unmarshalers, allow_missing_fields, allow_unknown_fields, load_filter,
            parents, attrib_path_eles[:-1], loaded_objects, currently_loading_object)
        return _JaxonLoadedFromReferenceWrapper(res)
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
        data = _load_data(group, attr_value, group_key_and_th, has_key_th)
        return dill.loads(data, **dill_kwargs)

    # handle container types
    debug_path = f"{debug_path}[{th}]"
    types = th.split("#")
    if types[0] == "dict":
        sub_group = group[group_key_and_th]
        pytree = {}
        dict_key_index, dict_key = None, None
        for i, sub_group_key in enumerate(sub_group.attrs):
            if sub_group_key.startswith(JAXON_DICT_KEY):
                if dict_key_index is not None:
                    raise JaxonError(f"expected {JAXON_DICT_KEY}({i}) to be "
                        f"followed immediately by {JAXON_DICT_VALUE}({i}) while parsing {debug_path!r}")
                dict_key_index = _parse_key_or_val(sub_group_key, JAXON_DICT_KEY)
                if len(pytree) != dict_key_index:
                    raise JaxonError(f"group key index error on {debug_path!r}")
                dbgstr = f"{debug_path}.key({i})"
                dict_key = _load(sub_group, sub_group_key, allow_dill, dill_kwargs,
                    dbgstr, custom_unmarshalers, allow_missing_fields, allow_unknown_fields,
                    load_filter, parents + [_DICT_KEY_PATH_ELEMENT], hdf5_path, loaded_objects,
                    currently_loading_object)
                continue
            if sub_group_key.startswith(JAXON_DICT_VALUE):
                index_in_value_key = _parse_key_or_val(sub_group_key, JAXON_DICT_VALUE)
                if dict_key_index is None or index_in_value_key != dict_key_index:
                    raise JaxonError(f"expected {JAXON_DICT_VALUE}({i}) to be followed immediately by "
                        f"{JAXON_DICT_KEY}({i}) while parsing {debug_path!r}")
                dict_key_index = None
            else:
                # assume that the key is a simple atom (fully represented by sub_group_key)
                if dict_key_index is not None:
                    raise JaxonError("did not expect presence of a "
                        f"{JAXON_DICT_KEY}({i}) while parsing {debug_path!r}")
                sub_group_key_data, _ = _get_group_key_and_typehint(sub_group_key)
                is_simple_atom, dict_key = _simple_atom_from_data_str(sub_group_key_data)
                if not is_simple_atom:
                    raise JaxonError(f"expected simple atom for sub group key {sub_group_key!r}")
            dbgstr = f"{debug_path}.{_key_to_debugstring(dict_key, i)}"
            pytree[dict_key] = _load(sub_group, sub_group_key, allow_dill, dill_kwargs,
                dbgstr, custom_unmarshalers, allow_missing_fields, allow_unknown_fields,
                load_filter, parents + [dict_key], hdf5_path, loaded_objects,
                currently_loading_object)
    elif types[0] in ("list", "tuple", "set", "frozenset"):
        sub_group = group[group_key_and_th]
        pytree = [_load(sub_group, sub_group_key, allow_dill, dill_kwargs, f"{debug_path}({i})",
                        custom_unmarshalers, allow_missing_fields, allow_unknown_fields,
                        load_filter, parents + [i], hdf5_path, loaded_objects,
                        currently_loading_object)
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
        new_pytree = _unmarshal_custom_obj(qualified_name, pytree, custom_unmarshalers,
            allow_missing_fields, allow_unknown_fields)
        if new_pytree is None:
            raise ValueError(f"cannot load custom object at {debug_path!r}, as type identified by "
                             f"{qualified_name!r} has no custom unmarshaler and its instance does "
                             "not has the from_jaxon method and is not a dataclass")
        pytree = new_pytree
    return pytree


def save(path_or_file, pytree: PyTree,
         exact_python_numeric_types: bool = True,
         downcast_to_base_types: Iterable | None = None,
         py_to_np_types: Iterable | None = None,
         allow_dill: bool = False,
         dill_kwargs: dict | None = None,
         storage_hints: Iterable[tuple[Any, JaxonStorageHints]] | None = None,
         custom_marshalers: tuple[Marshaler, ...] = tuple()) -> None:
    """
    Save a pytree in a human readable format in an hd5f file with the specified path or
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
        implicitly to ``np.int64``, ``np.float64``, ``np.bool`` and ``np.complex128`` respectively
        and stored as the corresponding hd5f binary type. If the file is loaded, the types will
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
                         py_to_np_types, custom_marshalers, tuple(), "", {})
    if hasattr(path_or_file, "seek") and hasattr(path_or_file, "truncate"):
        # when a file like object is provided
        # the file must be truncated like this because the "w"
        # mode does not seem to do this (bug?)
        path_or_file.seek(0)
        path_or_file.truncate()
    with h5py.File(path_or_file, 'w', track_order=True) as file:
        _store_atom(file, root_atom, JAXON_ROOT_GROUP_KEY, storage_hints_converted, {}, "/")


def load(path_or_file, allow_dill: bool = False, dill_kwargs: dict | None = None,
         custom_unmarshalers: Iterable[Unmarshaler] = tuple(),
         allow_missing_fields: bool = False, allow_unknown_fields: bool = False,
         load_filter: LoadFilter | None = None) -> PyTree:
    """
    Load a pytree from an hd5f file. It must be in the format produced by the ``save`` function.

    Parameters
    ----------
    path_or_file :
        A path-like object indicating the file path or a file-like object to read from.
        Providing a path-like object is the preferred option if possible (see the h5py
        documentation).
    allow_dill : bool, default=False
        Whether to allow loading objects serialized with ``dill``. If a serialized object is
        encountered and this argument is ``False``, an error is raised. 
    dill_kwargs : dict or None, optional
        Extra keyword arguments passed to ``dill.loads`` if ``allow_dill`` is True.
    custom_unmarshalers : Iterable[Unmarshaler]
        If provided, each custom type (identified by its qualified name) is passed
        as the first argument and its marshalled data (in the form of a python standard
        container or another custom object) as the second argument to the the Callables
        in the order they are provided. The return type shall be either ``None`` indicating
        that the Callable cannot unmarshal the type or a ``PyTree`` representing the
        successfully unmarshaled object. The first result that is not ``None`` is used.
        If all Callables return ``None``, the object is unmarshaled using the ``from_jaxon``
        interface (if available) or the default implementation for dataclasses.
    allow_missing_fields: bool, default=False
        Do not raise an error if fields are present in the hdf5 file which do not have a
        corresponding definition in the instanciated dataclass.
    allow_unknown_fields: bool, default=False
        Do not raise an error if fields are defined in a dataclass but are not found in
        the hdf5 file. The fields will be initialized using their default_factory or default
        value if available. Otherwise, they will be initialized with the constant
        ``JAXON_NOT_LOADED``.
    load_filter: LoadFilter or None
        If provided, the Callable controls what should be loaded. For each leaf or node in
        the pytree, it is called with a list of items that represent the path in the pytree as
        an argument and shall return ``True`` if the node or leaf shall be loaded and ``False``
        otherwise. For dictionaries the path element is the loaded dict key object, for lists
        and set like objects it is the index of the element (of type ``int``) and for
        dataclasses it is the field name. If the pytree node or leaf is not loaded, it is
        replaced with the constant ``JAXON_NOT_LOADED``.
    """
    with h5py.File(path_or_file, 'r') as file:
        # a type hint might have been added to the JAXON_ROOT_GROUP_KEY
        group_key = next((group_key for group_key in file.attrs
                          if group_key.startswith(JAXON_ROOT_GROUP_KEY)), None)
        if group_key is None:
            raise JaxonError("jaxon root group not found")
        if dill_kwargs is None:
            dill_kwargs = {}
        if load_filter is None:
            load_filter = lambda path: True
        return _load(file, group_key, allow_dill, dill_kwargs, "",
            tuple(custom_unmarshalers), allow_missing_fields, allow_unknown_fields,
            load_filter, [], tuple(), {}, set())
