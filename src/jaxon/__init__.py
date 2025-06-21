"""
Main module that provides the save and load functions, as well as to_jaxon, from_jaxon.

Author
------
Frank Hermann
"""

import jax.numpy as jnp
import numpy as np
import h5py
import dill


JAXON_NONE = "None"
JAXON_ROOT_NAME = "root"
JAXON_SCALAR_BINARY_TYPES = (np.int64, np.float64, np.bool, np.complex128)


def get_package_path(obj):
    return type(obj).__module__ + "." + type(obj).__qualname__


def create_instance(package_path):
    return None


def to_jaxon(pytree, group, name=JAXON_ROOT_NAME, allow_dill=False, dill_kwargs=None,
             exact_python_types=False, parent_objects=None, debug_path=None):
    if debug_path is None:
        debug_path = ""
    else:
        debug_path += "." + name
    
    # handle primitive scalar types
    if pytree is None:
        group.attrs[name] = JAXON_NONE
        return
    if exact_python_types and type(pytree) in (int, float, bool, complex):
        rep = repr(pytree)
        if type(pytree) is complex:
            rep = rep[1:-1]  # remove unnecessary brackets
        group.attrs[name] = f"{type(pytree).__name__}({rep})"
        return
    if (type(pytree) in JAXON_SCALAR_BINARY_TYPES
        or (not exact_python_types and isinstance(pytree, (int, float, bool, complex)))):
        # this is more relaxed than if exact_python_types=True
        # since it can implicitly converts int to int64 and so on
        group.attrs[name] = pytree
        return
    if type(pytree) is str or (not exact_python_types and isinstance(pytree, str)):
        # add qoutation marks to avoid possible naming collision with a type hint
        group.attrs[name] = repr(pytree)
        return

    # handle arrays
    if type(pytree) in (np.ndarray, jnp.ndarray, bytes):
        group.create_dataset(name, data=pytree)
        group.attrs[name] = "bytes" if type(pytree) is bytes else get_package_path(pytree)
        return

    # handle container types
    if parent_objects is None:
        parent_objects = [pytree]  # root node
    elif any(pytree is p for p in parent_objects):
        raise ValueError(f"detected circular reference in pytree at {debug_path!r}")
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
    if type(pytree) in (list, tuple, dict):
        attr_value = type(pytree).__name__ + attr_value
        if used_to_jaxon:
            debug_path += f"({attr_value})"
        sub_group = group.create_group(name, track_order=True)
        if type(pytree) is dict:
            for k, v in pytree.items():
                to_jaxon(v, sub_group, k, allow_dill, dill_kwargs, exact_python_types,
                            parent_objects, f"{debug_path}[{k!r}]")
        else:
            for i, v in enumerate(pytree):
                to_jaxon(v, sub_group, str(i), allow_dill, dill_kwargs, exact_python_types,
                        parent_objects, f"{debug_path}[{i}]")
        group.attrs[name] = attr_value
        return

    # use dill for any other objects if enabled
    attr_value = "!" + get_package_path(pytree) + attr_value  # the '!' denotes that the object is serialized
    if used_to_jaxon:
        debug_path += f"({attr_value})"
    if allow_dill:
        if dill_kwargs is None:
            dill_kwargs = {}
        group.create_dataset(name, data=dill.dumps(pytree, **dill_kwargs))
        group.attrs[name] = attr_value + "!" + get_package_path(pytree)
        return
    if used_to_jaxon:
        raise TypeError(f"Object at {debug_path!r} is not a valid jaxon container type; it was "
                         "returned by a to_jaxon method, but is not an instance of dict, list, "
                         "tuple or another object with a to_jaxon method. It can be serialized "
                         "anyway if allow_dill is set to True.")
    raise TypeError(f"Object at {debug_path!r} is not a valid jaxon type, but it can be "
                     "serialized if allow_dill is set to True.")


def from_jaxon(group, name, allow_dill=False, dill_kwargs=None, debug_path=None):
    if debug_path is None:
        debug_path = ""
    else:
        debug_path += "." + name
    val = group.attrs[name]

    # handle primitive scalar types
    if val == JAXON_NONE:
        return None
    if isinstance(val, JAXON_SCALAR_BINARY_TYPES):
        # here we also unpack primitives like int or float if they were saved
        # with exact_python_types=False
        return val
    assert type(val) is str, "unexpected hdf5 attribute type"
    if val[0] == "'":
        assert len(val) >= 2 and val[-1] == "'", "string parsing error: unexpected termination"
        return val[1:-1]
    for primitive in (int, float, bool, complex):
        # here, we parse primitives that were saved with exact_python_types=True
        type_name = primitive.__name__
        if val.startswith(type_name):
            assert val[len(type_name)] == "(" and val[-1] == ")", "primitive parsing error"
            return primitive(val[len(type_name) + 1:-1])

    # handle arrays
    if val in ("numpy.ndarray", "bytes"):
        return group[name][()]
    if val == "jax.ndarray":
        return jnp.array(group[name][()])

    # handle container types
    debug_path = f"{debug_path}({val})"
    types = val.split("#")
    if types[0] == "dict":
        dict_group = group[name]
        jaxon = {name: from_jaxon(dict_group, name, allow_dill, dill_kwargs, debug_path)
                 for name in dict_group.attrs}
    elif types[0] in ("list", "tuple"):
        dict_group = group[name]
        jaxon = [from_jaxon(dict_group, name, allow_dill, dill_kwargs, debug_path)
                 for name in dict_group.attrs]
        if types[0] == "tuple":
            jaxon = tuple(jaxon)
    elif types[0][0] == "!":
        if allow_dill:
            jaxon = dill.loads(group[name], **dill_kwargs)
        else:
            raise ValueError(f"cannot load serialized object at {debug_path!r}, "
                              "as allow_dill=False")
    else:
        raise ValueError(f"type of object at {debug_path!r} not undesrtood")
    for package_path in types[1:]:
        instance = create_instance(package_path)
        if hasattr(instance):
            jaxon = instance.from_jaxon(jaxon)
        else:
            raise ValueError(f"cannot load object at {debug_path!r}, as type "
                             f"{get_package_path!r} has not attribute from_jaxon")
    return jaxon


def save(path, pytree, exact_python_types=True, allow_dill=False, dill_kwargs=None):
    with h5py.File(path, 'w', track_order=True) as file:
        to_jaxon(pytree, file, exact_python_types=exact_python_types, allow_dill=allow_dill,
                 dill_kwargs=dill_kwargs)


def load(path, allow_dill=False, dill_kwargs=None):
    with h5py.File(path, 'r') as file:
        return from_jaxon(file, JAXON_ROOT_NAME, allow_dill=allow_dill, dill_kwargs=dill_kwargs)
