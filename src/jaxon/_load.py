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


from typing import Iterable
import importlib
import warnings
import dataclasses
import numpy as np
import jax
import h5py
import dill
from ._common import (
    JAXON_JAX_ARRAY, JAXON_NUMPY_ARRAY, JAXON_NUMPY_STR, JAXON_NUMPY_BYTES,
    JAXON_NUMPY_VOID, JaxonNotLoaded, JAXON_NUMPY_NUMERIC_TYPES, PyTree,
    Unmarshaler, LoadFilter, _PathElement,
    JAXON_NONE, JAXON_ELLIPSIS, JaxonFormatError,
    JAXON_DICT_KEY, JAXON_DICT_VALUE, JAXON_ROOT_GROUP_KEY,
    JaxonFormatWarning, JaxonError, CircularPyTreeException,
    _DICT_KEY_PATH_ELEMENT, _JAXON_MISSING,
    _JaxonLoadedFromReferenceWrapper, JaxonBuiltin,
    _get_qualified_name, _key_to_debugstring, JAXON_TRUE, JAXON_FALSE
)


def _get_class(qualified_name: str):
    """Get the class identified by `qualified_name` that was returned by `_get_qualified_name`."""
    parts = qualified_name.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _split_type_hint_and_type_arg(type_info: str) -> tuple[str, str | None]:
    type_info = type_info.strip()  # normally this should already be stripped
    if not type_info:
        return "", None
    if type_info[-1] == "]":
        if "]" in type_info[:-1]:
            raise JaxonFormatError("invalid type argument syntax (multiple occurrence of ']')")
        res = type_info[:-1].split("[")
        if len(res) > 2:
            raise JaxonFormatError("invalid type argument syntax (multiple occurrence of '[')")
        if len(res) < 2:
            raise JaxonFormatError("invalid type argument syntax (found ']', but no matching '[')")
        return res[0].rstrip(), res[1].lstrip()
    if "[" in type_info:
        raise JaxonFormatError("invalid type argument syntax (found '[', but no matching ']')")
    return type_info, None


def _supply_dataclass_unmarshaler(allow_missing_fields: bool,
        allow_unknown_fields: bool) -> Unmarshaler:
    def dataclass_unmarshaler(type_info: str, dict_: PyTree) -> PyTree | None:
        try:
            th, qualified_name = _split_type_hint_and_type_arg(type_info)
        except JaxonFormatError:
            return None
        if th != "dclass":
            return None
        if qualified_name is None:
            raise JaxonFormatError("expected type argument for dataclass")
        if type(dict_) is not dict:
            raise JaxonFormatError("expected dict for marshaled dataclass")
        cls_ = _get_class(qualified_name)
        if not dataclasses.is_dataclass(cls_):
            raise JaxonFormatError(f"expected that {qualified_name} is a dataclass")
        instance = cls_.__new__(cls_)
        fields = dataclasses.fields(instance)
        field_names = {f.name for f in fields}
        available_field_names = dict_.keys()
        missing_fields = available_field_names - field_names
        if missing_fields:
            message = f"The following fields in {_get_qualified_name(instance)!r} are present in " \
                    "the hdf5 file but are missing in the class " \
                    f"definition: {', '.join(missing_fields)}"
            if allow_missing_fields:
                warnings.warn(message, JaxonFormatWarning)
            else:
                raise ValueError(message + ". To omit this error run "
                    "with allow_missing_fields=True.")
        unknown_fields = field_names - available_field_names
        if unknown_fields:
            message = f"the following fields in {_get_qualified_name(instance)!r} are present in " \
                    "the class definition but are missing in the hdf5 " \
                    f"file: {', '.join(unknown_fields)}"
            if allow_unknown_fields:
                warnings.warn(message, JaxonFormatWarning)
            else:
                raise ValueError(message + ". To omit this error run with "
                    "allow_unknown_fields=True. Missing fields will be initialized using the "
                    "default_factory or default value. If both are missing JaxonNotLoaded() "
                    "will be used as a placeholder. The __post_init__() logic is never triggered.")
        for field in fields:
            try:
                val = dict_[field.name]
            except KeyError:
                if field.default_factory is not dataclasses.MISSING:
                    val = field.default_factory()
                elif field.default is not dataclasses.MISSING:
                    val = field.default
                else:
                    val = JaxonNotLoaded()
            # use object.__setattr__ to make it work even if the dataclass is frozen
            object.__setattr__(instance, field.name, val)
        return instance
    return dataclass_unmarshaler


def _jaxon_interface_unmarshaler(type_info: str, marshaled: PyTree) -> PyTree | None:
    try:
        th, qualified_name = _split_type_hint_and_type_arg(type_info)
    except JaxonFormatError:
        return None
    if th != "jaxon":
        return None
    if qualified_name is None:
        raise JaxonFormatError("expected type argument for dataclass")
    cls_ = _get_class(qualified_name)
    if not hasattr(cls_, 'from_jaxon'):
        raise JaxonFormatError(f"expected that the class {qualified_name!r} marshaled via the "
                               "to_jaxon method has the from_jaxon method")
    instance = cls_.__new__(cls_)
    instance.from_jaxon(marshaled)
    return instance


def _supply_dill_unmarshaler(allow_dill: bool, dill_kwargs: dict) -> Unmarshaler:
    def dill_unmarshaler(type_info: str, marshaled: PyTree) -> PyTree | None:
        try:
            th, ta = _split_type_hint_and_type_arg(type_info)
        except JaxonFormatError:
            return None
        if th != "dill":
            return None
        if ta is not None:
            raise JaxonFormatError("did a type argument for dill")
        if not allow_dill:
            raise ValueError("cannot load dill serialized object as allow_dill=False")
        return dill.loads(marshaled, **dill_kwargs)
    return dill_unmarshaler


def _unescape_str_terminator(string: str) -> str:
    return string.replace("\\'", "'").replace("\\\\", "\\")


def _simple_atom_from_data_str(data: str) -> PyTree:
    """Parse `data` and return the represented python object
    if `data` is Jaxon's string representation of `None`, `Ellipsis`,
    `bool`, `str`, `int`, `float`, `complex` or `range`.
    If `data` can not be interpreted as any of these types,
    return `_JAXON_MISSING`."""
    data = data.strip()
    if data == JAXON_NONE:
        return None
    if data == JAXON_ELLIPSIS:
        return Ellipsis
    if data == JAXON_TRUE:
        return True
    if data == JAXON_FALSE:
        return False
    if data[0] == "'":
        if len(data) < 2 and data[-1] != "'":
            raise JaxonFormatError(f"string {data!r} not terminated")
        return _unescape_str_terminator(data[1:-1])
    try:
        return int(data)
    except ValueError:
        pass
    try:
        return float(data)
    except ValueError:
        pass
    try:
        return complex(data)
    except ValueError:
        pass
    if data.startswith("range"):
        if not data.startswith("range(") or data[-1] != ")":
            raise ValueError(f"cannot parse range representation: {data!r}")
        parts = data[len("range("):-1].split(",")
        if len(parts) not in (2, 3):
            raise ValueError(f"cannot parse range representation: {data!r}")
        try:
            return range(*[int(f) for f in parts])
        except ValueError as e:
            raise ValueError(f"cannot parse range representation: {data!r}") from e
    return _JAXON_MISSING


def _split_attrib_name_and_type_info(attrib_key: str) -> tuple[str, str]:
    """Separates the attribute name from a possibly added typehint.
    The returned strings are stripped."""
    for i in reversed(range(len(attrib_key))):
        ch = attrib_key[i]
        if ch == "'":
            # string begins (cannot happen in the typehint;
            # therefore there is no typehint)
            break
        if ch == ":":
            attrib_name = attrib_key[:i]
            typehint = attrib_key[i + 1:]
            return attrib_name.strip(), typehint.strip()
    return attrib_key.strip(), ""


def _parse_key_or_val(group_key: str, prefix: str) -> int:
    if group_key[len(prefix)] != "(":
        raise JaxonError(f"malformed group key {group_key!r}")
    return int(group_key[len(prefix) + 1:group_key.find(")")])


def _tokenize_attrib_path(path: str) -> tuple[str]:
    """
    Return a tuple of the stripped path elements of the reference path.
    
    This function effectively splits the string at all `/` characters. The provided
    path must start with a `/`. The first returned path element is the string after
    the leading slash till the next slash, the last one is the string after the last
    `/` character. Slash characters inside string literals are ignored in the process.
    Path elements are stripped and returned even if they are empty strings.
    Unterminated string literals will raise a `JaxonFormatError`. This function does
    not perform any unescaping of string literals. It also does not check if escape
    sequences in string literals are valid.
    """
    assert path[0] == "/"
    buf = ""
    res = []
    in_string = False
    escape_next_char = False
    for c in path[1:]:
        if in_string:
            if escape_next_char:
                escape_next_char = False
            else:
                if c == "'":
                    in_string = False
                elif c == "\\":
                    escape_next_char = True
        else:
            assert not escape_next_char
            if c == "/":
                res.append(buf.strip())
                buf = ""
                continue
            if c == "'":
                in_string = True
        buf += c
    res.append(buf.strip())
    if in_string:
        raise JaxonFormatError(f"string literal not terminated in path {path!r}")
    assert not escape_next_char
    return tuple(res)


def _get_fixed_length_string(group, attrib_key, attrib_value, debug_path: str) -> str:
    info = h5py.check_string_dtype(group.attrs.get_id(attrib_key).dtype)
    if info is None or info.length is None or info.encoding != "utf-8":
        raise JaxonError(f"expected fixed length string for HDF5 attribute at {debug_path!r}")
    return attrib_value.decode("utf-8").strip()


def _load_dataset(group, dataset_name: str):
    return group[dataset_name][()]


def _assert_no_type_arg(ta: str | None, th: str, debug_path: str) -> None:
    if ta is not None:
        raise JaxonFormatError(f"did not expect a type argument for type {th!r} at {debug_path!r}")


def _assert_type_arg(ta: str | None, th: str, debug_path: str) -> None:
    if ta is None:
        raise JaxonFormatError(f"expected a type argument for type {th!r} at {debug_path!r}")


def _load_builtin_type(group, attrib_key: str, attrib_name: str, type_info: str, debug_path: str,
                       unmarshalers: tuple[Unmarshaler, ...], load_filter: LoadFilter,
                       parents: list[_PathElement], group_path: str,
                       loaded_objects: dict[str, PyTree],
                       currently_loading_object: set[str]) \
                           -> JaxonBuiltin | _JaxonLoadedFromReferenceWrapper:
    th, ta = _split_type_hint_and_type_arg(type_info)
    if not th:
        attrib_value = group.attrs[attrib_key]
        if type(attrib_value) in JAXON_NUMPY_NUMERIC_TYPES:
            return attrib_value
        attrib_value_str = _get_fixed_length_string(group, attrib_key, attrib_value, debug_path)
        pytree = _simple_atom_from_data_str(attrib_value_str)
        if pytree is not _JAXON_MISSING:
            return pytree
        # if it's not a simple atom, it must be a reference
        if attrib_value_str[0] == "/":
            attrib_path_eles = _tokenize_attrib_path(attrib_value_str)
            target_group = group.file
            group_path = "/"
            for group_key in attrib_path_eles[:-1]:
                target_group = target_group[group_key]
                group_path += group_key + "/"
            res = _load(target_group, attrib_path_eles[-1], debug_path, unmarshalers, load_filter,
                parents, group_path, loaded_objects, currently_loading_object)
            return _JaxonLoadedFromReferenceWrapper(res)
        raise JaxonFormatError("can not interpret the stringified "
            f"data in HDF5 attribute {debug_path!r}")
    if th == JAXON_JAX_ARRAY:
        _assert_type_arg(ta, th, debug_path)
        return jax.numpy.asarray(_load_dataset(group, attrib_name), dtype=ta)
    if th == "bytes":
        _assert_no_type_arg(ta, th, debug_path)
        res = _load_dataset(group, attrib_name)
        if isinstance(res, h5py.Empty):
            return b''
        return bytes(res)
    if th == "bytearray":
        _assert_no_type_arg(ta, th, debug_path)
        res = _load_dataset(group, attrib_name)
        if isinstance(res, h5py.Empty):
            return bytearray()
        return bytearray(res)
    if th == JAXON_NUMPY_ARRAY:
        _assert_no_type_arg(ta, th, debug_path)
        # np.asarray needed to correctly restore 0d arrays
        return np.asarray(_load_dataset(group, attrib_name))
    if th == JAXON_NUMPY_STR:
        _assert_no_type_arg(ta, th, debug_path)
        res = _load_dataset(group, attrib_name)
        if isinstance(res, h5py.Empty):
            return np.str_()
        if type(res) is not np.bytes_:
            res = np.bytes_(res)
        return np.str_(res.decode("utf-8"))
    if th == JAXON_NUMPY_BYTES:
        _assert_no_type_arg(ta, th, debug_path)
        res = _load_dataset(group, attrib_name)
        if isinstance(res, h5py.Empty):
            return np.bytes_()
        if type(res) is not np.bytes_:
            res = np.bytes_(res)
        return res
    if th == JAXON_NUMPY_VOID:
        _assert_no_type_arg(ta, th, debug_path)
        res = _load_dataset(group, attrib_name)
        if isinstance(res, h5py.Empty):
            return np.void(b'')
        return res

    # handle container types
    group_key = attrib_name  # same assumption as in saver
    sub_group_path = group_path + group_key + "/"
    debug_path = f"{debug_path}[{th}]"
    if th == "dict":
        _assert_no_type_arg(ta, th, debug_path)
        sub_group = group[group_key]
        pytree = {}
        dict_key_index, dict_key = None, None
        for i, sub_group_key in enumerate(sub_group.attrs):
            sub_group_key = sub_group_key.strip()
            if sub_group_key.startswith(JAXON_DICT_KEY):
                if dict_key_index is not None:
                    raise JaxonError(f"expected {JAXON_DICT_KEY}({i}) to be "
                        f"followed immediately by {JAXON_DICT_VALUE}({i}) "
                        f"while parsing {debug_path!r}")
                dict_key_index = _parse_key_or_val(sub_group_key, JAXON_DICT_KEY)
                if len(pytree) != dict_key_index:
                    raise JaxonError(f"group key index error on {debug_path!r}")
                dbgstr = f"{debug_path}.key({i})"
                dict_key = _load(sub_group, sub_group_key, dbgstr, unmarshalers,
                    load_filter, parents + [_DICT_KEY_PATH_ELEMENT], sub_group_path,
                    loaded_objects, currently_loading_object)
                continue
            if sub_group_key.startswith(JAXON_DICT_VALUE):
                index_in_value_key = _parse_key_or_val(sub_group_key, JAXON_DICT_VALUE)
                if dict_key_index is None or index_in_value_key != dict_key_index:
                    raise JaxonError(f"encountered {JAXON_DICT_VALUE}({index_in_value_key}) "
                        f"without a preceding matching {JAXON_DICT_KEY} while "
                        f"parsing {debug_path!r}")
                dict_key_index = None
            else:
                # assume that the key is a simple atom (fully represented by sub_group_key)
                if dict_key_index is not None:
                    raise JaxonError(f"expected {JAXON_DICT_VALUE}({dict_key_index}) but got "
                        f"simple key attribute {sub_group_key!r} while parsing {debug_path!r}")
                sub_group_key_data, _ = _split_attrib_name_and_type_info(sub_group_key)
                dict_key = _simple_atom_from_data_str(sub_group_key_data)
                if dict_key is _JAXON_MISSING:
                    raise JaxonError(f"expected simple atom for sub group key {sub_group_key!r}")
            dbgstr = f"{debug_path}.{_key_to_debugstring(dict_key, i)}"
            pytree[dict_key] = _load(sub_group, sub_group_key, dbgstr, unmarshalers,
                load_filter, parents + [dict_key], sub_group_path, loaded_objects,
                currently_loading_object)
        return pytree
    if th in ("list", "tuple", "set", "frozenset"):
        _assert_no_type_arg(ta, th, debug_path)
        sub_group = group[group_key]
        pytree = [_load(sub_group, sub_group_key, f"{debug_path}({i})", unmarshalers, load_filter,
                        parents + [i], sub_group_path, loaded_objects, currently_loading_object)
                  for i, sub_group_key in enumerate(sub_group.attrs)]
        if th == "tuple":
            pytree = tuple(pytree)
        if th == "set":
            pytree = set(pytree)
        if th == "frozenset":
            pytree = frozenset(pytree)
        return pytree
    raise ValueError(f"base type of object at {debug_path!r} not understood")


def _build_data_path(group_path: str, attrib_name: str, sub_type_info: list[str]) -> str:
    data_path = group_path + attrib_name
    if sub_type_info != [""]:
        data_path += ":" + "#".join(sub_type_info)
    return data_path


def _load(group,
          attrib_key: str,
          debug_path: str,
          unmarshalers: tuple[Unmarshaler, ...],
          load_filter: LoadFilter, parents: list[_PathElement],
          group_path: str,
          loaded_objects: dict[str, PyTree],
          currently_loading_attribs: set[str]) -> PyTree:
    assert group_path[-1] == "/"

    # check if the user requested loading of the object
    if not any(p is _DICT_KEY_PATH_ELEMENT for p in parents) and not load_filter(parents):
        return JaxonNotLoaded()

    # parse attribute key
    attrib_name, type_info = _split_attrib_name_and_type_info(attrib_key)
    sub_type_info = [s.strip() for s in type_info.split("#")]

    # Check if the attribute has already been loaded.
    # This happens if loading was triggered earlier by a reference.
    # Note that either the entire attribute is loaded
    # (with all intermediate results from the marshaler) or nothing is loaded.
    full_data_path = _build_data_path(group_path, attrib_name, sub_type_info)
    pytree = loaded_objects.get(full_data_path)
    if pytree is not None:
        return pytree

    # protected against circular references, which would lead to infinite recursion
    attrib_path = group_path + attrib_name
    if attrib_path in currently_loading_attribs:
        raise CircularPyTreeException()
    currently_loading_attribs.add(attrib_path)

    # load the object, if not already loaded
    pytree = _load_builtin_type(group, attrib_key, attrib_name, sub_type_info[0], debug_path,
        unmarshalers, load_filter, parents, group_path, loaded_objects,
        currently_loading_attribs)
    if isinstance(pytree, _JaxonLoadedFromReferenceWrapper):
        # indicates that the result was loaded from a reference
        # since references to references are not allowed in jaxon
        # it is not necessary to add the path to loaded_objects
        pytree = pytree.pytree
    else:
        data_path = _build_data_path(group_path, attrib_name, sub_type_info[:1])
        loaded_objects[data_path] = pytree
    for i, marshaler_type_info in enumerate(sub_type_info[1:]):
        for unmarshaler in unmarshalers:
            result = unmarshaler(marshaler_type_info, pytree)
            if result is not None:
                pytree = result
                break
        else:
            raise ValueError(f"cannot load object at {debug_path!r}, as type identified by "
                             f"{marshaler_type_info!r} has no suitable unmarshaler")
        # register the intermediate result as it might be required by a reference later
        data_path = _build_data_path(group_path, attrib_name, sub_type_info[:i + 2])
        loaded_objects[data_path] = pytree

    # done loading object
    currently_loading_attribs.remove(attrib_path)
    return pytree


def load(path_or_file,
         allow_dill: bool = False,
         dill_kwargs: dict | None = None,
         custom_unmarshalers: Iterable[Unmarshaler] = tuple(),
         allow_missing_fields: bool = False,
         allow_unknown_fields: bool = False,
         load_filter: LoadFilter | None = None) -> PyTree:
    """
    Load a pytree from an HDF5 file. It must be in the format produced by the ``save`` function.

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
        corresponding definition in the instantiated dataclass.
    allow_unknown_fields: bool, default=False
        Do not raise an error if fields are defined in a dataclass but are not found in
        the hdf5 file. The fields will be initialized using their default_factory or default
        value if available. Otherwise, they will be initialized with an instance of
        ``JaxonNotLoaded``.
    load_filter: LoadFilter or None
        If provided, the Callable controls what should be loaded. For each leaf or node in
        the pytree, it is called with a list of items that represent the path in the pytree as
        an argument and shall return ``True`` if the node or leaf shall be loaded and ``False``
        otherwise. For dictionaries the path element is the loaded dict key object, for lists
        and set like objects it is the index of the element (of type ``int``) and for
        dataclasses it is the field name. If the pytree node or leaf is not loaded, it is
        replaced with an instance of ``JaxonNotLoaded``. Note: dict keys are always loaded
        regardless of the filter — without them the dict cannot be reconstructed.
    """
    with h5py.File(path_or_file, 'r') as file:
        # a type hint might have been added to the JAXON_ROOT_GROUP_KEY
        attrib_key = next((attrib_key for attrib_key in file.attrs
                          if attrib_key.startswith(JAXON_ROOT_GROUP_KEY)), None)
        if attrib_key is None:
            raise JaxonError("jaxon root group not found")
        if dill_kwargs is None:
            dill_kwargs = {}
        if load_filter is None:
            def no_filter(_):
                return True
            load_filter = no_filter
        unmarshalers = list(custom_unmarshalers)
        unmarshalers += [
            _jaxon_interface_unmarshaler,
            _supply_dataclass_unmarshaler(allow_missing_fields, allow_unknown_fields),
            _supply_dill_unmarshaler(allow_dill, dill_kwargs)
        ]
        return _load(file, attrib_key, "", tuple(unmarshalers), load_filter, [], "/", {}, set())
