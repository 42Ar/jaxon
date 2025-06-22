"""
Main module that provides the save and load functions, as well as to_jaxon and from_jaxon.

Author
------
Frank Hermann
"""

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


class CircularPytreeException(Exception):
    """Raised when a circular reference (reference to a parent object) was detected."""
    pass


def get_package_path(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def create_instance(package_path):
    return None


def range_from_repr(repr_content):
    fields = repr_content.split(",")
    assert len(fields) in (2, 3)
    return range(*[int(f) for f in fields])


def slice_from_repr(repr_content):
    fields = repr_content.split(",")
    assert len(fields) == 3
    return slice(*[(None if f == "None" else int(f)) for f in fields])


def _base_type_name(obj, types, downcast_to_base_types):
    for t in types:
        if type(obj) is t or (type(obj) in downcast_to_base_types and isinstance(obj, t)):
            return t.__name__
    return None


def to_jaxon(pytree, group, name=JAXON_ROOT_NAME, allow_dill=False, dill_kwargs=None,
             downcast_to_base_types: tuple = tuple(), py_to_np_types: tuple = tuple(),
             parent_objects=None, debug_path=""):
    # handle primitive scalar types
    if pytree is None:
        group.attrs[name] = JAXON_NONE
        return
    if pytree is ...:
        group.attrs[name] = JAXON_ELLIPSIS
        return
    np_numeric_type = _base_type_name(pytree, JAXON_NP_NUMERIC_TYPES, downcast_to_base_types)
    if np_numeric_type is not None:
        group.attrs[name] = pytree
        return
    py_numeric_type = _base_type_name(pytree, JAXON_PY_NUMERIC_TYPES, downcast_to_base_types)
    if py_numeric_type is not None:
        if isinstance(pytree, py_to_np_types):
            # this assignment implicitly converts the type from python to numpy (e.g. int to int64)
            group.attrs[name] = pytree
            return
        rep = repr(pytree)
        if isinstance(pytree, complex):
            rep = rep[1:-1]  # remove unnecessary brackets
        group.attrs[name] = f"{py_numeric_type}({rep})"
        return
    other_repr_type = _base_type_name(pytree, (str, range, slice), downcast_to_base_types)
    if other_repr_type is not None:
        # note: repr adds qoutation marks for str which avoids possible naming
        # collision with a type hint
        attr_value = repr(pytree)
        if isinstance(pytree, (range, slice)):
            # remove unnecessary spaces which would cause parsing to fail
            attr_value = attr_value.replace(" ", "")
        group.attrs[name] = attr_value
        return

    # handle arrays
    array_type = _base_type_name(pytree, (np.ndarray, jnp.ndarray), downcast_to_base_types)
    if array_type is not None and pytree.dtype in JAXON_NP_NUMERIC_TYPES:
        group.create_dataset(name, data=pytree)
        group.attrs[name] = "numpy.ndarray" if isinstance(pytree, np.ndarray) else "jax.ndarray"
        return
    byte_buffer_type = _base_type_name(pytree, (bytes, bytearray, memoryview), downcast_to_base_types)
    if byte_buffer_type is not None:
        # do not store as a (fixed or variable) string, as this causes errors with zero bytes at the end
        group.create_dataset(name, data=np.frombuffer(pytree, dtype=np.uint8))
        group.attrs[name] = byte_buffer_type
        return

    # handle container types
    if parent_objects is None:
        parent_objects = [pytree]  # root node
    elif any(pytree is p for p in parent_objects):
        raise CircularPytreeException(f"detected circular reference in pytree at {debug_path!r}")
    else:
        parent_objects = parent_objects + [pytree]  # Descend. Need new list of parents here.
    used_to_jaxon = False
    attr_value = ""
    while hasattr(pytree, "to_jaxon"):
        used_to_jaxon = True
        pytree = pytree.to_jaxon()
        # the '#' indicates that the class uses the to_jaxon/from_jaxon interface
        attr_value = attr_value + "#" + get_package_path(pytree)
        parent_objects += [pytree]
    container_type = _base_type_name(pytree, (list, tuple, dict, set, frozenset), downcast_to_base_types)
    if container_type is not None:
        attr_value = container_type + attr_value
        debug_path += f"[{attr_value}]"
        sub_group = group.create_group(name, track_order=True)
        if isinstance(pytree, dict):
            for k, v in pytree.items():
                to_jaxon(v, sub_group, k, allow_dill, dill_kwargs, downcast_to_base_types,
                         py_to_np_types, parent_objects, f"{debug_path}.{k}")
        else:
            for i, v in enumerate(pytree):
                to_jaxon(v, sub_group, str(i), allow_dill, dill_kwargs, downcast_to_base_types,
                         py_to_np_types, parent_objects, f"{debug_path}({i})")
        group.attrs[name] = attr_value
        return

    # use dill for any other objects if enabled
    attr_value = "!" + get_package_path(pytree) + attr_value  # the '!' denotes that the object is serialized
    debug_path += f"[{attr_value}]"
    if allow_dill:
        if dill_kwargs is None:
            dill_kwargs = {}
        pybytes = dill.dumps(pytree, **dill_kwargs)
        # do not store as a (fixed or variable) string, as this causes errors with zero bytes at the end
        group.create_dataset(name, data=np.frombuffer(pybytes, dtype=np.uint8))
        group.attrs[name] = attr_value
        return
    if used_to_jaxon:
        raise TypeError(f"Object at {debug_path!r} is not a valid jaxon container type; it was "
                         "returned by a to_jaxon method, but is not an instance of dict, list, "
                         "tuple or another object with a to_jaxon method. It can be serialized "
                         "anyway if allow_dill is set to True.")
    raise TypeError(f"Object at {debug_path!r} is not a valid jaxon type, but it can be "
                     "serialized if allow_dill is set to True.")


def from_jaxon(group, name, allow_dill=False, dill_kwargs=None, debug_path=""):
    val = group.attrs[name]

    # handle primitive scalar types
    if val == JAXON_NONE:
        return None
    if val == JAXON_ELLIPSIS:
        return ...
    if isinstance(val, JAXON_NP_NUMERIC_TYPES):
        # here we also unpack primitives like int or float if they were saved
        # with exact_python_types=False
        return val
    assert type(val) is str, "unexpected hdf5 attribute type"
    if val[0] == "'":
        assert len(val) >= 2 and val[-1] == "'", "string parsing error: unexpected termination"
        return val[1:-1]
    other_repr_types = [(int, None), (float, None), (bool, None), (complex, None),
                        (range, range_from_repr), (slice, slice_from_repr)]
    for primitive, parser in other_repr_types:
        # here, we parse primitives that were saved with exact_python_types=True
        type_name = primitive.__name__
        if not val.startswith(type_name):
            continue
        assert val[len(type_name)] == "(" and val[-1] == ")", "primitive parsing error"
        repr_content = val[len(type_name) + 1:-1]
        if parser is not None:
            return parser(repr_content)
        return primitive(repr_content)

    # handle arrays
    if val == "bytes":
        return bytes(group[name][()])
    if val == "bytearray":
        return bytearray(group[name][()])
    if val == "memoryview":
        return memoryview(group[name][()])
    if val == "numpy.ndarray":
        return group[name][()]
    if val == "jax.ndarray":
        return jnp.array(group[name][()])

    # handle container types
    debug_path = f"{debug_path}[{val}]"
    types = val.split("#")
    if types[0] == "dict":
        dict_group = group[name]
        jaxon = {name: from_jaxon(dict_group, name, allow_dill, dill_kwargs, f"{debug_path}.{name}")
                 for name in dict_group.attrs}
    elif types[0] in ("list", "tuple", "set", "frozenset"):
        dict_group = group[name]
        jaxon = [from_jaxon(dict_group, name, allow_dill, dill_kwargs, f"{debug_path}({i})")
                 for i, name in enumerate(dict_group.attrs)]
        if types[0] == "tuple":
            jaxon = tuple(jaxon)
        if types[0] == "set":
            jaxon = set(jaxon)
        if types[0] == "frozenset":
            jaxon = frozenset(jaxon)
    elif types[0][0] == "!":
        if allow_dill:
            if dill_kwargs is None:
                dill_kwargs = {}
            jaxon = dill.loads(group[name][()], **dill_kwargs)
        else:
            raise ValueError(f"cannot load serialized object at {debug_path!r}, "
                              "as allow_dill=False")
    else:
        raise ValueError(f"type of object at {debug_path!r} not understood")
    for package_path in types[1:]:
        instance = create_instance(package_path)
        if hasattr(instance):
            jaxon = instance.from_jaxon(jaxon)
        else:
            raise ValueError(f"cannot load object at {debug_path!r}, as type "
                             f"{get_package_path!r} has not attribute from_jaxon")
    return jaxon


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
    with h5py.File(path, 'w', track_order=True) as file:
        to_jaxon(pytree, file, downcast_to_base_types=downcast_to_base_types,
                 py_to_np_types=py_to_np_types, allow_dill=allow_dill, dill_kwargs=dill_kwargs)


def load(path, allow_dill=False, dill_kwargs=None):
    with h5py.File(path, 'r') as file:
        return from_jaxon(file, JAXON_ROOT_NAME, allow_dill=allow_dill, dill_kwargs=dill_kwargs)
