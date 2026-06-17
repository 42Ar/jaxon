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
HDF5 reading and atom-to-pytree reconstruction (load side).
"""


from typing import Any, Iterable
import importlib
import warnings
import dataclasses
import numpy as np
import jax
import h5py
import dill

from ._common import (
    PyTree, Unmarshaler, LoadFilter, _PathElement,
    JAXON_NP_NUMERIC_TYPES, JAXON_NONE, JAXON_ELLIPSIS,
    JAXON_DICT_KEY, JAXON_DICT_VALUE, JAXON_ROOT_GROUP_KEY, JAXON_REF,
    JaxonFormatWarning, JaxonError, CircularPyTreeException,
    JAXON_NOT_LOADED, _DICT_KEY_PATH_ELEMENT, _JAXON_MISSING,
    _JaxonLoadedFromReferenceWrapper,
    _get_qualified_name, _key_to_debugstring,
)


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


def _decode_string(buffer):
    """All string are stored as utf-8 fixed length strings."""
    return buffer.decode("utf-8")


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
                  f"definition: {', '.join(missing_fields)}"
        if allow_missing_fields:
            warnings.warn(message, JaxonFormatWarning)
        else:
            raise ValueError(message + ". To omit this error run with allow_missing_fields=True.")
    unknown_fields = field_names - available_field_names
    if unknown_fields:
        message = f"the following fields in {_get_qualified_name(instance)!r} are present in " \
                   "the class definition but are missing in the hdf5 " \
                  f"file: {', '.join(unknown_fields)}"
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


def _do_load(group, group_key_and_th: str, allow_dill: bool, dill_kwargs: dict,
             debug_path: str, custom_unmarshalers: tuple[Unmarshaler, ...],
             allow_missing_fields: bool, allow_unknown_fields: bool,
             load_filter: LoadFilter, parents: list[_PathElement], hdf5_path: tuple[str, ...],
             loaded_objects: dict[tuple[str, ...], PyTree],
             currently_loading_object: set[tuple[str, ...]]) -> PyTree:
    """Recursively load the pytree from the hd5f file. Here, `group` is an h5py group object,
    the `group_key_and_th` is the group key (including a possible type hint) which must be
    a valid key in the group's attribute dict."""
    # dict keys are never filtered: _DICT_KEY_PATH_ELEMENT in parents means we are currently
    # loading a dict key, which must always be loaded so the dict can be reconstructed.
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
        replaced with the constant ``JAXON_NOT_LOADED``. Note: dict keys are always loaded
        regardless of the filter — without them the dict cannot be reconstructed.
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
