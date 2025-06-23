"""
Main module that provides the save and load functions.

Author
------
Frank Hermann
"""

from typing import Any
from dataclasses import dataclass, field
import dataclasses
import importlib
import jax
import numpy as np
import h5py
import dill


JAXON_ROOT_GROUP_KEY = "JAXON_ROOT"
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
JAXON_CONTAINER_TYPES = (list, tuple, dict, set, frozenset)
JAXON_JAX_ARRAY_TYPE = type(jax.numpy.array([]))  # get the type of a jax array (in a version-independent way)


class CircularPytreeException(Exception):
    """Raised when a circular reference (reference to a parent object) was detected."""
    pass


@dataclass
class JaxonDict:
    data: list[tuple['JaxonAtom', 'JaxonAtom']] = field(default_factory=list)


@dataclass
class JaxonList:
    data: list['JaxonAtom'] = field(default_factory=list)


JAXON_TYPES = (JaxonDict, JaxonList, *JAXON_PY_NUMERIC_TYPES, JAXON_NP_NUMERIC_TYPES, memoryview, np.ndarray, str)


@dataclass
class JaxonAtom:
    data: Any
    typehint: str | None = None
    original_obj_id: int | None = None

    def _is_simple(self) -> bool:
        """A simple atom encodes the data and typehint only into in the data field
        which must be a str that does not contain null chars. This means that
        simple atoms can be used as group or attribute keys in the hd5f file."""
        return self.typehint is None and type(self.data) is str and "\0" not in self.data


@dataclass
class JaxonStorageHints:
    store_in_dataset: bool


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


def _dataclass_to_container(instance):
    return {field.name: getattr(instance, field.name) for field in dataclasses.fields(instance)}


def _custom_obj_to_container(pytree):
    if hasattr(pytree, "to_jaxon"):
        return True, pytree.to_jaxon()
    if dataclasses.is_dataclass(pytree):
        return True, _dataclass_to_container(pytree)
    return False, None


def _container_to_dataclass(container, instance):
    assert type(container) is dict, "expected dict container for dataclass"
    for field_name, field_value in container.items():
        setattr(instance, field_name, field_value)


def _container_to_custom_obj(container, instance):
    if hasattr(instance, "from_jaxon"):
        instance.from_jaxon(container)
    elif dataclasses.is_dataclass(instance):
        _container_to_dataclass(container, instance)
    else:
        return False
    return True


def to_atom(pytree, allow_dill=False, dill_kwargs=None, downcast_to_base_types: tuple = tuple(),
             py_to_np_types: tuple = tuple(), parent_objects=None, debug_path="") -> JaxonAtom:
    atom = _to_atom(pytree, allow_dill, dill_kwargs, downcast_to_base_types, py_to_np_types,
                    parent_objects, debug_path)
    return JaxonAtom(atom.data, atom.typehint, id(pytree))


def _to_atom(pytree, allow_dill, dill_kwargs, downcast_to_base_types, py_to_np_types,
             parent_objects, debug_path) -> JaxonAtom:
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
        # collision with a type hint
        typehint = repr(pytree)
        if isinstance(pytree, (range, slice)):
            # remove unnecessary spaces which would cause parsing to fail
            typehint = typehint.replace(" ", "")
        return JaxonAtom(typehint)
    str_type = _base_type_name(pytree, (str,), downcast_to_base_types)
    if str_type is not None:
        # add qoutation marks to avoid possible naming collision with type hint
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
                value_atom = to_atom(dict_value, allow_dill, dill_kwargs, downcast_to_base_types,
                                     py_to_np_types, parent_objects, f"{debug_path}.{key_atom.typehint}")
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

    # use dill for any other types if enabled
    typehint = "!" + _get_qualified_name(pytree) + typehint  # the '!' denotes that the object is serialized
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
    if atom.typehint is None:
        _store_in_attrib(group, atom.data, group_key)
    elif isinstance(atom.data, JaxonDict):
        sub_group = group.create_group(group_key, track_order=True)
        for i, (key_atom, value_atom) in enumerate(atom.data.data):
            if key_atom._is_simple():
                group_key_of_value = key_atom.data
            else:
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
            _store_in_attrib(group, atom.data, f"{group_key}:{atom.typehint}")            
        else:
            _store_in_attrib(group, atom.typehint, group_key)
            group.create_dataset(group_key, data=atom.data)
    else:
        assert False, f"unexpected internal jaxon data type {type(atom.data)!r}"


def _simple_atom_from_value(typehint_or_data):
    # handle primitive scalar types
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
    # attention must be paid here as the colons (which seperate the typehint)
    # are not escped in strings
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


def load_data(group, attr_value, group_key_with_th, has_key_th):
    if has_key_th:
        # presence of a type hint in the key implies that the data resides
        # in the attribute value (load it it if it's not already loaded)
        if attr_value is None:
            return group.attrs[group_key_with_th]
        return attr_value 
    else:
        # otherwise, it resides in a dataset
        return group[group_key_with_th][()]


def _load(group, group_key_and_th, allow_dill=False, dill_kwargs=None, debug_path=""):
    _, th = _get_group_key_and_typehint(group_key_and_th)
    has_key_th = th is not None
    attr_value = None  # if None, it will be loaded later on demand (if necesseray)
    if not has_key_th:
        attr_value = group.attrs[group_key_and_th]
        if type(attr_value) in JAXON_NP_NUMERIC_TYPES:
            # here we also laod primitives like int or float if they were saved
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
        is_simple_atom, pytree = _simple_atom_from_value(th_or_data)
        if is_simple_atom:
            return pytree
        # if it's not a simple atom, it must be a typehint
        th = th_or_data

    # handle arrays
    if th == "bytes":
        return bytes(load_data(group, attr_value, group_key_and_th, has_key_th))
    if th == "bytearray":
        return bytearray(load_data(group, attr_value, group_key_and_th, has_key_th))
    if th == "memoryview":
        return memoryview(load_data(group, attr_value, group_key_and_th, has_key_th))
    if th == "numpy.ndarray":
        return load_data(group, attr_value, group_key_and_th, has_key_th)
    if th == "jax.Array":
        return jax.numpy.array(load_data(group, attr_value, group_key_and_th, has_key_th))

    # handle serialized types
    if th[0] == "!":
        if allow_dill:
            if dill_kwargs is None:
                dill_kwargs = {}
            data = load_data(group, attr_value, group_key_and_th, has_key_th)
            return dill.loads(data, **dill_kwargs)
        else:
            raise ValueError(f"cannot load serialized object at {debug_path!r}, "
                              "as allow_dill=False")

    # handle container types
    debug_path = f"{debug_path}[{th}]"
    types = th.split("#")
    if types[0] == "dict":
        sub_group = group[group_key_and_th]
        pytree = {}
        for sub_group_key in sub_group.attrs:
            if sub_group_key.startswith("value"):
                continue  # loaded if corresponing key is read
            if sub_group_key.startswith("key"):
                assert sub_group_key[len("key")] == "(" and sub_group_key[-1] == ")"
                group_key_of_value = f"value({int(sub_group_key[len('key')+1:-1])})"
                key = _load(sub_group, sub_group_key, allow_dill, dill_kwargs, f"{debug_path}.{sub_group_key}")
            else:
                # assume that the key is a simple atom (fully represented by sub_group_key)
                sub_group_key_data, _ = _get_group_key_and_typehint(sub_group_key)  # discard typehint
                is_simple_atom, key = _simple_atom_from_value(sub_group_key_data)
                assert is_simple_atom, f"expected simple atom for sub group key {sub_group_key!r}"
                group_key_of_value = sub_group_key
            pytree[key] = _load(sub_group, group_key_of_value, allow_dill, dill_kwargs, f"{debug_path}.{key}")
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


def save(path, pytree, exact_python_numeric_types=True, downcast_to_base_types=None,
         py_to_np_types=None, allow_dill=False, dill_kwargs=None,
         storage_hints: list[tuple[Any, JaxonStorageHints]] = None):
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
        storage_hints = {}
    else:
        storage_hints = {id(obj): hint for obj, hint in storage_hints}
    root_atom = to_atom(pytree, allow_dill, dill_kwargs, downcast_to_base_types, py_to_np_types)
    with h5py.File(path, 'w', track_order=True) as file:
        _store_atom(file, root_atom, JAXON_ROOT_GROUP_KEY, storage_hints)


def load(path, allow_dill=False, dill_kwargs=None):
    with h5py.File(path, 'r') as file:
        # a type hint might have been added to the JAXON_ROOT_GROUP_KEY
        group_key = next((group_key for group_key in file.attrs if group_key.startswith(JAXON_ROOT_GROUP_KEY)), None)
        assert group_key is not None, "jaxon root group not found"
        return _load(file, group_key, allow_dill=allow_dill, dill_kwargs=dill_kwargs)

