"""
Main module that provides the save and load functions.

Author
------
Frank Hermann
"""

from typing import Any
from dataclasses import dataclass, field
import importlib
import jax.numpy as jnp
import numpy as np
import h5py
import dill


JAXON_ROOT_NAME = "root"
JAXON_NP_NUMERIC_TYPES = (
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64, np.float128,
    np.complex64, np.complex128,
    np.bool)
JAXON_PY_NUMERIC_TYPES = (int, float, bool, complex)
JAXON_NONE = "None"
JAXON_ELLIPSIS = "Ellipsis"
JAXON_DICT_KEY = "key"
JAXON_DICT_VALUE = "value"


class CircularPytreeException(Exception):
    """Raised when a circular reference (reference to a parent object) was detected."""
    pass


@dataclass
class JaxonAtom:
    value: Any
    data: Any = None


@dataclass
class JaxonDict:
    data: list[tuple[JaxonAtom, JaxonAtom]] = field(default_factory=list)


@dataclass
class JaxonList:
    data: list[JaxonAtom] = field(default_factory=list)


def _get_qualified_name(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def _create_instance(qualified_name: str):
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


def _to_atom(pytree, allow_dill=False, dill_kwargs=None, downcast_to_base_types: tuple = tuple(),
             py_to_np_types: tuple = tuple(), parent_objects=None, debug_path="") -> JaxonAtom:
    # handle primitive scalar types
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
        # collision with a type hint
        value = repr(pytree)
        if isinstance(pytree, (range, slice)):
            # remove unnecessary spaces which would cause parsing to fail
            value = value.replace(" ", "")
        return JaxonAtom(value)
    str_type = _base_type_name(pytree, (str,), downcast_to_base_types)
    if str_type is not None:
        # add qoutation marks to avoid possible naming collision with type hint
        return JaxonAtom("'" + pytree + "'")

    # handle arrays
    array_type = _base_type_name(pytree, (np.ndarray, jnp.ndarray), downcast_to_base_types)
    if array_type is not None and pytree.dtype in JAXON_NP_NUMERIC_TYPES:
        if isinstance(pytree, jnp.ndarray):
            return JaxonAtom("jax.numpy.ndarray", np.array(pytree))
        else:
            return JaxonAtom("numpy.ndarray", pytree)
    byte_buffer_type = _base_type_name(pytree, (bytes, bytearray, memoryview), downcast_to_base_types)
    if byte_buffer_type is not None:
        return JaxonAtom(byte_buffer_type, pytree if isinstance(pytree, memoryview) else memoryview(pytree))

    # handle container types first
    if parent_objects is None:
        parent_objects = [pytree]  # root node
    elif any(pytree is p for p in parent_objects):
        raise CircularPytreeException(f"detected circular reference in pytree at {debug_path!r}")
    else:
        parent_objects = parent_objects + [pytree]  # Descend. Need new list of parents here.
    used_to_jaxon = False
    value = ""
    while hasattr(pytree, "to_jaxon"):
        used_to_jaxon = True
        # the '#' indicates that the class uses the to_jaxon/from_jaxon interface
        value = "#" + _get_qualified_name(pytree) + value
        pytree = pytree.to_jaxon()
        parent_objects += [pytree]
    container_type = _base_type_name(pytree, (list, tuple, dict, set, frozenset), downcast_to_base_types)
    if container_type is not None:
        value = container_type + value
        debug_path += f"[{value}]"
        if isinstance(pytree, dict):
            data = JaxonDict()
            for i, (dict_key, dict_value) in enumerate(pytree.items()):
                key_atom = _to_atom(dict_key, allow_dill, dill_kwargs, downcast_to_base_types,
                                    py_to_np_types, parent_objects, f"{debug_path}.key({i})")
                value_atom = _to_atom(dict_value, allow_dill, dill_kwargs, downcast_to_base_types,
                                      py_to_np_types, parent_objects, f"{debug_path}.{key_atom.value}")
                data.data.append((key_atom, value_atom))
        else:
            data = JaxonList()
            for i, item in enumerate(pytree):
                item_atom = _to_atom(item, allow_dill, dill_kwargs, downcast_to_base_types,
                                     py_to_np_types, parent_objects, f"{debug_path}({i})")
                data.data.append(item_atom)
        return JaxonAtom(value, data)
    if used_to_jaxon:
        raise TypeError(f"Object at {debug_path!r} is not a valid jaxon container type; it was "
                         "returned by a to_jaxon method, but is not an instance of dict, list, "
                         "tuple, set or frozenset or another object with a to_jaxon method.")

    # use dill for any other types if enabled
    value = "!" + _get_qualified_name(pytree) + value  # the '!' denotes that the object is serialized
    debug_path += f"[{value}]"
    if allow_dill:
        if dill_kwargs is None:
            dill_kwargs = {}
        return JaxonAtom(value, memoryview(dill.dumps(pytree, **dill_kwargs)))
    raise TypeError(f"Object at {debug_path!r} is not a valid jaxon type, but it can be "
                     "serialized if allow_dill is set to True.")


def _store_atom_value(group, value, group_key):
    if isinstance(value, str):
        group.attrs[group_key] = _encode_string(value)
    elif isinstance(value, (*JAXON_PY_NUMERIC_TYPES, *JAXON_NP_NUMERIC_TYPES)):
        group.attrs[group_key] = value
    else:
        assert False, f"unexpected internal jaxon value type {type(value)!r}"


def _is_simple_atom(atom):
    """A simple atom has no additional data and is therefore fully represented
    by it's value that must be a string. Furthermore, the value cannot contain
    null chars, so that the value can be used as a group or attribute or dataset
    key in the hd5f file."""
    return isinstance(atom.value, str) and atom.data is None and "\0" not in atom.value


def _store_atom_data(group, data, group_key):
    if isinstance(data, JaxonDict):
        sub_group = group.create_group(group_key, track_order=True)
        for i, (key_atom, value_atom) in enumerate(data.data):
            if _is_simple_atom(key_atom):
                group_key_of_value = key_atom.value
            else:
                group_key_of_value = f"{JAXON_DICT_VALUE}({i})"
                group_key_of_key = f"{JAXON_DICT_KEY}({i})"
                _store_atom(sub_group, key_atom, group_key_of_key)
            _store_atom(sub_group, value_atom, group_key_of_value)
    elif isinstance(data, JaxonList):
        sub_group = group.create_group(group_key, track_order=True)
        for i, item_atom in enumerate(data.data):
            _store_atom(sub_group, item_atom, str(i))
    elif isinstance(data, (np.ndarray, memoryview)):
        group.create_dataset(group_key, data=data)
    else:
        assert False, f"unexpected internal jaxon data type {type(data)!r}"


def _store_atom(group, atom, key):
    _store_atom_value(group, atom.value, key)
    if atom.data is not None:
        _store_atom_data(group, atom.data, key)


def _simple_atom_from_value(value):
    # handle primitive scalar types
    if value == JAXON_NONE:
        return True, None
    if value == JAXON_ELLIPSIS:
        return True, ...
    if value[0] == "'":
        assert len(value) >= 2 and value[-1] == "'", "string parsing error: unexpected termination"
        return True, value[1:-1]
    other_repr_types = [(int, None), (float, None), (bool, None), (complex, None),
                        (range, _range_from_repr), (slice, _slice_from_repr)]
    for primitive, parser in other_repr_types:
        # here, we parse primitives that were saved with exact_python_types=True
        type_name = primitive.__name__
        if not value.startswith(type_name):
            continue
        assert value[len(type_name)] == "(" and value[-1] == ")", "primitive parsing error"
        repr_content = value[len(type_name) + 1:-1]
        if parser is not None:
            return True, parser(repr_content)
        return True, primitive(repr_content)
    return False, None


def _load(group, group_key, allow_dill=False, dill_kwargs=None, debug_path=""):
    value = group.attrs[group_key]
    if type(value) in JAXON_NP_NUMERIC_TYPES:
        # here we also unpack primitives like int or float if they were saved
        # with exact_python_types=False
        return value
    attr_dtype = group.attrs.get_id(group_key).dtype
    string_dtype = h5py.check_string_dtype(attr_dtype)
    assert string_dtype is not None, "unexpected hdf5 attribute type"
    assert string_dtype.length is not None, "expected a fixed length string"
    assert string_dtype.encoding == "utf-8", "unexpected string encoding"
    value = _decode_string(value)

    # handle simple atoms first
    is_simple_atom, pytree = _simple_atom_from_value(value)
    if is_simple_atom:
        return pytree
    
    # handle arrays
    if value == "bytes":
        return bytes(group[group_key][()])
    if value == "bytearray":
        return bytearray(group[group_key][()])
    if value == "memoryview":
        return memoryview(group[group_key][()])
    if value == "numpy.ndarray":
        return group[group_key][()]
    if value == "jax.numpy.ndarray":
        return jnp.array(group[group_key][()])

    # handle serialized types
    if value[0] == "!":
        if allow_dill:
            if dill_kwargs is None:
                dill_kwargs = {}
            return dill.loads(group[group_key][()], **dill_kwargs)
        else:
            raise ValueError(f"cannot load serialized object at {debug_path!r}, "
                              "as allow_dill=False")

    # handle container types
    debug_path = f"{debug_path}[{value}]"
    types = value.split("#")
    if types[0] == "dict":
        sub_group = group[group_key]
        pytree = {}
        for sub_group_key in sub_group.attrs:
            if sub_group_key.startswith("value"):
                continue  # loaded if corresponing key is read
            if sub_group_key.startswith("key"):
                assert sub_group_key[len("key")] == "(" and sub_group_key[-1] == ")"
                group_key_of_value = f"value({int(sub_group_key[len('key')+1:-1])})"
                key = _load(sub_group, sub_group_key, allow_dill, dill_kwargs, f"{debug_path}.{sub_group_key}")
            else:
                is_simple_atom, key = _simple_atom_from_value(sub_group_key)
                assert is_simple_atom, f"expected simple atom for sub group key {sub_group_key!r}"
                group_key_of_value = sub_group_key
            pytree[key] = _load(sub_group, group_key_of_value, allow_dill, dill_kwargs, f"{debug_path}.{key}")
        return pytree
    elif types[0] in ("list", "tuple", "set", "frozenset"):
        sub_group = group[group_key]
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
        if hasattr(instance, "from_jaxon"):
            instance.from_jaxon(pytree)
            pytree = instance
        else:
            raise ValueError(f"cannot load object at {debug_path!r}, as type "
                             f"{_get_qualified_name!r} has not attribute from_jaxon")
    return pytree


def save(path, pytree, exact_python_numeric_types=True, downcast_to_base_types=None,
         py_to_np_types=None, allow_dill=False, dill_kwargs=None):
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
    root_atom = _to_atom(pytree, allow_dill, dill_kwargs, downcast_to_base_types, py_to_np_types)
    with h5py.File(path, 'w', track_order=True) as file:
        _store_atom(file, root_atom, JAXON_ROOT_NAME)


def load(path, allow_dill=False, dill_kwargs=None):
    with h5py.File(path, 'r') as file:
        return _load(file, JAXON_ROOT_NAME, allow_dill=allow_dill, dill_kwargs=dill_kwargs)
