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
This module handles HDF5 reading and pytree reconstruction.
"""


from typing import Iterable, Union
import importlib
import warnings
import dataclasses
import numpy as np
import jax
import h5py
import dill
from ._common import (
    JAXON_JAX_ARRAY, JAXON_NUMPY_ARRAY, JAXON_NUMPY_STR, JAXON_NUMPY_BYTES,
    JAXON_NUMPY_VOID, JaxonNotLoaded, JAXON_NUMPY_NUMERIC_TYPES, JaxonNumpyNumeric,
    PyTree, Unmarshaler, LoadFilter, PathElement, JAXON_NONE, JAXON_ELLIPSIS,
    JaxonFormatError, JAXON_DICT_KEY, JAXON_DICT_VALUE, JAXON_ROOT_GROUP_KEY,
    JaxonFormatWarning, JaxonError, CircularPyTreeError, DICT_KEY_PATH_ELEMENT,
    JAXON_MISSING, JaxonBuiltin, get_qualified_name, JAXON_TRUE, JAXON_FALSE,
    JaxonMissing
)


class AttribInfo:
    """
    Represents an HDF5 attribute. Caches loaded pytrees and holds meta information.
    
    Attributes
    ----------
    group: HDF5Group
        Group in which this attribute is contained.
    attrib_key: str
        Full attribute key (including possible whitespace).
        Must be a valid attribute key in the HDF5 group.
        For example ` mydict : dict#MyCustomClass `.
    attrib_name: str
        Stripped attribute name, without any type information.
        For example `mydict`.
    type_info: str
        Full type information. For example `dict#MyCustomClass`.
    loaded_pytree: PyTree | None
        If `JAXON_MISSING`, the attribute has not been loaded yet.
        Otherwise, contains references to the loaded pytree.
    is_loading: bool
        If `True` this attribute is currently being loaded.
    """
    group: "GroupInfo"
    attrib_key: str
    attrib_name: str
    type_info: str
    loaded_pytree: PyTree | JaxonMissing
    is_loading: bool

    def __init__(self, group: "GroupInfo", attrib_key: str, attrib_name: str, type_info: str):
        self.group = group
        self.attrib_key = attrib_key
        self.attrib_name = attrib_name
        self.type_info = type_info
        self.loaded_pytree = JAXON_MISSING
        self.is_loading = False

    def get_value(self):
        """Get the raw attribute value from HDF5. Not cached."""
        return self.group.group.attrs[self.attrib_key]

    def get_fixed_length_string(self, attrib_value) -> str:
        """Make sure that attrib_value is a fixed length utf-8 string and decode it."""
        info = h5py.check_string_dtype(self.group.group.attrs.get_id(self.attrib_key).dtype)
        if info is None or info.length is None or info.encoding != "utf-8":
            raise JaxonFormatError(f"expected fixed length string for attribute {self.path()!r}")
        return attrib_value.decode("utf-8").strip()

    def path(self) -> str:
        return self.group.path + self.attrib_key


class GroupInfo:
    """
    Represents an HDF5 group.
    
    Attributes
    ----------
    group: h5py.Group
        Reference to the h5py group.
    path: str
        Path in the HDF5 file.
    objects: dict[str, h5py.Group | h5py.Dataset | GroupInfo]
        Maps a stripped dataset or group name to either a `h5py.Dataset`,
        `h5py.Group` or `GroupInfo`. Populated by the constructor. If a group
        is requested, it is implicitly converted to an `GroupInfo`.
    attributes: dict[str, AttribInfo]
        Maps all stripped attribute names to their respective `AttribInfo`
        object. Populated by the constructor.
    root: GroupInfo
        Reference to the root group.
    """
    group: h5py.Group
    path: str
    parent: Union["h5py.Group", None]
    objects: dict[str, Union[h5py.Group, h5py.Dataset, "GroupInfo"]]
    attributes: dict[str, AttribInfo]
    root: "GroupInfo"

    def __init__(self, group: h5py.Group, path: str,
            root: Union["GroupInfo", None] = None):
        """Populates the internal objects and group attributes. If root is `None`
        it is assumed that this is the root group."""
        self.group = group
        self.path = path
        self.objects = {key.strip(): val for key, val in group.items()}
        self.attributes = {}
        if root is None:
            self.root = self
            for attrib_key in group.attrs:
                # a type hint might have been added to the JAXON_ROOT_GROUP_KEY
                if attrib_key.lstrip().startswith(JAXON_ROOT_GROUP_KEY):
                    self._add_attrib(attrib_key)
                    break
            else:
                raise JaxonFormatError("jaxon root attribute not found (key "
                                       f"must start with {JAXON_ROOT_GROUP_KEY!r})")
        else:
            self.root = root
            for attrib_key in group.attrs:
                self._add_attrib(attrib_key)

    def _add_attrib(self, attrib_key: str) -> None:
        attrib_name, type_info = split_attrib_name_and_type_info(attrib_key)
        self.attributes[attrib_name] = AttribInfo(self, attrib_key, attrib_name, type_info)

    def load_dataset(self, dataset_name: str):
        """Retrieve a dataset. The `dataset_name` must be a stripped string. Not cached."""
        dataset = self.objects[dataset_name]
        if not isinstance(dataset, h5py.Dataset):
            raise JaxonFormatError(f"expected a dataset for {dataset_name!r} in "
                                   f"group {self.path!r}")
        return dataset[()]

    def get_group(self, group_key: str) -> "GroupInfo":
        """Get the group identified by `group_key`, which
        is assumed to be a stripped string. Cached."""
        group = self.objects.get(group_key)
        if group is None:
            raise JaxonFormatError(f"group {group_key!r} not found in group {self.path!r}")
        if isinstance(group, GroupInfo):
            return group
        if isinstance(group, h5py.Group):
            group = GroupInfo(group, self.path + group_key + "/", self.root)
            self.objects[group_key] = group
            return group
        raise JaxonFormatError(f"object {group_key!r} in group {self.path!r} has unexpected "
                               "type, expected dataset")

    def get_attrib(self, attrib_key: str) -> AttribInfo:
        """"Retrieve an `AttribInfo` object. The `attrib_key` must be a stripped string."""
        attrib = self.attributes.get(attrib_key)
        if attrib is None:
            raise JaxonFormatError(f"attribute {attrib_key!r} not found in group {self.path!r}")
        return attrib


def get_class(qualified_name: str):
    """Get the class identified by `qualified_name` that was returned by `get_qualified_name`."""
    parts = qualified_name.split(".")
    module_path = ".".join(parts[:-1])
    class_name = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def split_type_hint_and_type_arg(type_info: str) -> tuple[str, str | None]:
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


def supply_dataclass_unmarshaler(allow_missing_fields: bool,
        allow_unknown_fields: bool) -> Unmarshaler:
    """Create an unmarshaler for dataclasses."""
    def dataclass_unmarshaler(type_info: str, dict_: PyTree) -> PyTree | None:
        try:
            th, qualified_name = split_type_hint_and_type_arg(type_info)
        except JaxonFormatError:
            return None
        if th != "dclass":
            return None
        if qualified_name is None:
            raise JaxonFormatError("expected type argument for dataclass")
        if type(dict_) is not dict:
            raise JaxonFormatError("expected dict for marshaled dataclass")
        cls_ = get_class(qualified_name)
        if not dataclasses.is_dataclass(cls_):
            raise JaxonFormatError(f"expected that {qualified_name!r} is a dataclass")
        instance = cls_.__new__(cls_)  # type: ignore
        fields = dataclasses.fields(instance)
        field_names = {f.name for f in fields}
        available_field_names = dict_.keys()
        missing_fields = available_field_names - field_names
        if missing_fields:
            message = f"The following fields in {get_qualified_name(instance)!r} are present in " \
                      "the HDF5 file but are missing in the class " \
                      f"definition: {', '.join(map(repr, missing_fields))}"
            if allow_missing_fields:
                warnings.warn(message, JaxonFormatWarning)
            else:
                raise JaxonFormatError(message + ". To omit this error run "
                                       "with allow_missing_fields=True.")
        unknown_fields = field_names - available_field_names
        if unknown_fields:
            message = f"the following fields in {get_qualified_name(instance)!r} are present in " \
                      "the class definition but are missing in the HDF5 " \
                      f"file: {', '.join(map(repr, unknown_fields))}"
            if allow_unknown_fields:
                warnings.warn(message, JaxonFormatWarning)
            else:
                raise JaxonFormatError(message + ". To omit this error run with "
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


def jaxon_interface_unmarshaler(type_info: str, marshaled: PyTree) -> PyTree | None:
    """Unmarshaler for objects that implement the `from_jaxon` method."""
    try:
        th, qualified_name = split_type_hint_and_type_arg(type_info)
    except JaxonFormatError:
        return None
    if th != "jaxon":
        return None
    if qualified_name is None:
        raise JaxonFormatError("expected type argument")
    cls_ = get_class(qualified_name)
    if not hasattr(cls_, 'from_jaxon'):
        raise JaxonFormatError(f"expected that the class {qualified_name!r} marshaled via the "
                               "to_jaxon method has the from_jaxon method")
    instance = cls_.__new__(cls_)
    instance.from_jaxon(marshaled)
    return instance


def supply_dill_unmarshaler(allow_dill: bool, dill_kwargs: dict) -> Unmarshaler:
    """Unmarshaler for objects that were marshaled with dill."""
    def dill_unmarshaler(type_info: str, marshaled: PyTree) -> PyTree | None:
        try:
            th, ta = split_type_hint_and_type_arg(type_info)
        except JaxonFormatError:
            return None
        if th != "dill":
            return None
        if ta is not None:
            raise JaxonFormatError("expected a type argument")
        if not allow_dill:
            raise JaxonFormatError("cannot load dill serialized object as allow_dill=False")
        return dill.loads(marshaled, **dill_kwargs)
    return dill_unmarshaler


def unescape_str_terminator(string: str) -> str:
    return string.replace("\\'", "'").replace("\\\\", "\\")


def simple_atom_from_data_str(data: str, attrib: AttribInfo) -> PyTree:
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
            raise JaxonFormatError(f"string {data!r} not terminated in "
                                   f"attribute {attrib.path()!r}")
        return unescape_str_terminator(data[1:-1])
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
            raise JaxonFormatError(f"cannot parse range representation: {data!r} "
                                   f"for attribute {attrib.path()!r}")
        parts = data[len("range("):-1].split(",")
        if len(parts) not in (2, 3):
            raise JaxonFormatError(f"cannot parse range representation: {data!r} "
                                   f"for attribute {attrib.path()!r}")
        try:
            return range(*[int(f) for f in parts])
        except ValueError as e:
            raise JaxonFormatError(f"cannot parse range representation: {data!r} "
                                   f"for attribute {attrib.path()!r}") from e
    return JAXON_MISSING


def split_attrib_name_and_type_info(attrib_key: str) -> tuple[str, str]:
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


def parse_key_or_val(group_key: str, prefix: str, group: GroupInfo) -> int:
    closing = group_key.find(")")
    if closing == -1 or len(group_key) <= len(prefix) or group_key[len(prefix)] != "(":
        raise JaxonError(f"malformed group key {group_key!r} in group {group!r}")
    return int(group_key[len(prefix) + 1:closing])


def tokenize_attrib_path(path: str) -> tuple[str]:
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


def assert_no_type_arg(ta: str | None, attrib: AttribInfo) -> None:
    if ta is not None:
        raise JaxonFormatError(f"did not expect a type argument for attrib {attrib.path()!r}")


def assert_type_arg(ta: str | None, attrib: AttribInfo) -> None:
    if ta is None:
        raise JaxonFormatError(f"expected a type argument for attribute {attrib.path()!r}")


def load_builtin_type(attrib: AttribInfo, type_info: str, parents: list[PathElement],
        unmarshalers: tuple[Unmarshaler, ...], load_filter: LoadFilter) -> JaxonBuiltin:
    """Recursively load the pytree represented by `attrib` iff the pytree is a jaxon builtin type.
    Otherwise, raise an error."""
    th, ta = split_type_hint_and_type_arg(type_info)
    if not th:
        attrib_value = attrib.get_value()
        if type(attrib_value) in JAXON_NUMPY_NUMERIC_TYPES:
            assert isinstance(attrib_value, JaxonNumpyNumeric)
            return attrib_value
        attrib_value_str = attrib.get_fixed_length_string(attrib_value)
        pytree = simple_atom_from_data_str(attrib_value_str, attrib)
        if pytree is not JAXON_MISSING:
            return pytree
        # if it's not a simple atom, it must be a reference
        if attrib_value_str[0] == "/":
            attrib_path_eles = tokenize_attrib_path(attrib_value_str)
            target_group = attrib.group.root
            for group_key in attrib_path_eles[:-1]:
                target_group = target_group.get_group(group_key)
            target_attrib = target_group.get_attrib(attrib_path_eles[-1])
            return do_load(target_attrib, parents, unmarshalers, load_filter)
        raise JaxonFormatError(f"can not interpret string data {attrib_value_str!r} "
                               f"of attribute {attrib.path()!r}")
    if th == JAXON_JAX_ARRAY:
        assert_type_arg(ta, attrib)
        return jax.numpy.asarray(attrib.group.load_dataset(attrib.attrib_name), dtype=ta)
    if th == "bytes":
        assert_no_type_arg(ta, attrib)
        res = attrib.group.load_dataset(attrib.attrib_name)
        if isinstance(res, h5py.Empty):
            return b''
        return bytes(res)
    if th == "bytearray":
        assert_no_type_arg(ta, attrib)
        res = attrib.group.load_dataset(attrib.attrib_name)
        if isinstance(res, h5py.Empty):
            return bytearray()
        return bytearray(res)
    if th == JAXON_NUMPY_ARRAY:
        assert_no_type_arg(ta, attrib)
        # np.asarray needed to correctly restore 0d arrays
        return np.asarray(attrib.group.load_dataset(attrib.attrib_name))
    if th == JAXON_NUMPY_STR:
        assert_no_type_arg(ta, attrib)
        res = attrib.group.load_dataset(attrib.attrib_name)
        if isinstance(res, h5py.Empty):
            return np.str_()
        if type(res) is not np.bytes_:
            res = np.bytes_(res)
        return np.str_(res.decode("utf-8"))
    if th == JAXON_NUMPY_BYTES:
        assert_no_type_arg(ta, attrib)
        res = attrib.group.load_dataset(attrib.attrib_name)
        if isinstance(res, h5py.Empty):
            return np.bytes_()
        if type(res) is not np.bytes_:
            res = np.bytes_(res)
        return res
    if th == JAXON_NUMPY_VOID:
        assert_no_type_arg(ta, attrib)
        res = attrib.group.load_dataset(attrib.attrib_name)
        if isinstance(res, h5py.Empty):
            return np.void(b'')
        return res

    # handle container types
    if th == "dict":
        assert_no_type_arg(ta, attrib)
        sub_group = attrib.group.get_group(attrib.attrib_name)
        pytree = {}
        index_in_key, dict_key = None, None
        for i, sub_attrib in enumerate(sub_group.attributes.values()):
            if sub_attrib.attrib_name.startswith(JAXON_DICT_KEY):
                if index_in_key is not None:
                    raise JaxonFormatError(f"expected '{JAXON_DICT_KEY}({i})' to be "
                        f"followed immediately by '{JAXON_DICT_VALUE}({i})' while loading "
                        f"{attrib.path()!r}")
                index_in_key = parse_key_or_val(sub_attrib.attrib_name, JAXON_DICT_KEY, sub_group)
                if len(pytree) != index_in_key:
                    raise JaxonFormatError(f"unexpected index {i} of key attribute while "
                                           f"loading {attrib.path()!r}")
                dict_key = do_load(sub_attrib, parents + [DICT_KEY_PATH_ELEMENT],
                                 unmarshalers, load_filter)
                continue
            if sub_attrib.attrib_name.startswith(JAXON_DICT_VALUE):
                index_in_value = parse_key_or_val(sub_attrib.attrib_name, JAXON_DICT_VALUE, sub_group)
                if index_in_key is None or index_in_value != index_in_key:
                    raise JaxonFormatError(f"encountered {JAXON_DICT_VALUE}({index_in_key}) "
                        f"without a preceding matching {JAXON_DICT_KEY}({index_in_key}) "
                        f"while loading {attrib.path()!r}")
                index_in_key = None
            else:
                if index_in_key is not None:
                    raise JaxonFormatError(f"expected '{JAXON_DICT_KEY}({i})' to be "
                        f"followed immediately by '{JAXON_DICT_VALUE}({i})' while loading "
                        f"{attrib.path()!r}")
                # assume that the key is a simple atom
                # (fully represented by sub_attrib.attrib_name)
                dict_key = simple_atom_from_data_str(sub_attrib.attrib_name, sub_attrib)
                if dict_key is JAXON_MISSING:
                    raise JaxonFormatError(f"expected that group key {sub_attrib.attrib_name!r} " \
                                           f"is a simple atom while loading {attrib.path()!r}")
            pytree[dict_key] = do_load(sub_attrib, parents + [dict_key], unmarshalers, load_filter)
        return pytree
    if th in ("list", "tuple", "set", "frozenset"):
        assert_no_type_arg(ta, attrib)
        sub_group = attrib.group.get_group(attrib.attrib_name)
        pytree = [do_load(sub_attrib, parents + [i], unmarshalers, load_filter)
                  for i, sub_attrib in enumerate(sub_group.attributes.values())]
        if th == "tuple":
            pytree = tuple(pytree)
        if th == "set":
            pytree = set(pytree)
        if th == "frozenset":
            pytree = frozenset(pytree)
        return pytree
    raise JaxonFormatError(f"type hint {th!r} not understood, expected a "
                           f"type hint for a jaxon builtin type while loading {attrib.path()!r}")


def do_load(attrib: AttribInfo, parents: list[PathElement],
            unmarshalers: tuple[Unmarshaler, ...], load_filter: LoadFilter) -> PyTree:
    """Load the pytree represented by `attrib`."""

    # check if the user requested loading of the object
    if not any(p is DICT_KEY_PATH_ELEMENT for p in parents) and not load_filter(parents):
        return JaxonNotLoaded()

    # check if the attribute has already been loaded
    if not isinstance(attrib.loaded_pytree, JaxonMissing):
        return attrib.loaded_pytree

    # protected against circular references, which would lead to infinite recursion
    if attrib.is_loading:
        raise CircularPyTreeError("detected circular reference "
                                  f"in pytree at {attrib.path()!r}")
    attrib.is_loading = True

    # parse type information
    type_info = [s.strip() for s in attrib.type_info.split("#")]

    # load the attribute
    # the type specified by type_info[0] must be a builtin type
    pytree = load_builtin_type(attrib, type_info[0], parents, unmarshalers, load_filter)

    # now apply the unmarshalers (if required) to the loaded builtin type
    for type_hint in type_info[1:]:
        for unmarshaler in unmarshalers:
            try:
                result = unmarshaler(type_hint, pytree)
            except Exception as e:
                e.args = (f"Exception occurred while applying unmarshaler "
                    f"{unmarshaler.__name__!r} to attribute {attrib.path()!r}: {e.args[0]}",) \
                    + e.args[1:]
                raise
            if result is not None:
                pytree = result
                break
        else:
            raise JaxonFormatError("no suitable unmarshaler found for typehint "
                f"{type_hint!r} to unmarshal {attrib.path()!r}")

    # done loading object
    attrib.is_loading = False
    attrib.loaded_pytree = pytree
    return pytree


def load(path_or_file,
         allow_dill: bool = False,
         dill_kwargs: dict | None = None,
         custom_unmarshalers: Iterable[Unmarshaler] = tuple(),
         allow_missing_fields: bool = False,
         allow_unknown_fields: bool = False,
         load_filter: LoadFilter | None = None) -> PyTree:
    """
    Load a pytree from an HDF5 file. It must be in the format produced by the `save` function.

    Parameters
    ----------
    path_or_file :
        A path-like object indicating the file path or a file-like object to read from.
        Providing a path-like object is the preferred option if possible (see the h5py
        documentation).
    allow_dill : bool, default=False
        Whether to allow loading objects serialized with `dill`. If a serialized object is
        encountered and this argument is `False`, an error is raised. 
    dill_kwargs : dict or None, optional
        Extra keyword arguments passed to `dill.loads` if `allow_dill` is True.
    custom_unmarshalers : Iterable[Unmarshaler]
        If provided, each custom type (identified by its qualified name) is passed
        as the first argument and its marshalled data (in the form of a python standard
        container or another custom object) as the second argument to the the Callables
        in the order they are provided. The return type shall be either `None` indicating
        that the Callable cannot unmarshal the type or a `PyTree` representing the
        successfully unmarshaled object. The first result that is not `None` is used.
        If all Callables return `None`, the object is unmarshaled using the `from_jaxon`
        interface (if available) or the default implementation for dataclasses.
    allow_missing_fields: bool, default=False
        Do not raise an error if fields are present in the HDF5 file which do not have a
        corresponding definition in the instantiated dataclass.
    allow_unknown_fields: bool, default=False
        Do not raise an error if fields are defined in a dataclass but are not found in
        the HDF5 file. The fields will be initialized using their default_factory or default
        value if available. Otherwise, they will be initialized with an instance of
        `JaxonNotLoaded`.
    load_filter: LoadFilter or None
        If provided, the Callable controls what should be loaded. For each leaf or node in
        the pytree, it is called with a list of items that represent the path in the pytree as
        an argument and shall return `True` if the node or leaf shall be loaded and `False`
        otherwise. For dictionaries the path element is the loaded dict key object, for lists
        and set like objects it is the index of the element (of type `int`) and for
        dataclasses it is the field name. If the pytree node or leaf is not loaded, it is
        replaced with an instance of `JaxonNotLoaded`. Note: dict keys are always loaded
        regardless of the filter — without them the dict cannot be reconstructed.
    """
    if dill_kwargs is None:
        dill_kwargs = {}
    if load_filter is None:
        def no_filter(_):
            return True
        load_filter = no_filter
    unmarshalers = list(custom_unmarshalers)
    unmarshalers += [
        jaxon_interface_unmarshaler,
        supply_dataclass_unmarshaler(allow_missing_fields, allow_unknown_fields),
        supply_dill_unmarshaler(allow_dill, dill_kwargs)
    ]
    with h5py.File(path_or_file, 'r') as file:
        root_group = GroupInfo(file, "/", None)
        root_attrib = root_group.get_attrib(JAXON_ROOT_GROUP_KEY)
        return do_load(root_attrib, [], tuple(unmarshalers), load_filter)
