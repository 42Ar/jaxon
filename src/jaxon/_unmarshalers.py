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
Built in unmarshalers to reconstruct custom marshaled objects.
"""

import dataclasses
import warnings
import dill
from ._common import (
    JaxonFormatError, JaxonFormatWarning, JaxonNotLoaded, PyTree, Unmarshaler, 
    get_class, get_qualified_name, is_namedtuple, split_type_hint_and_type_arg
)


def _check_fields(
    qualified_name: str,
    class_field_names: set[str],
    loaded_field_names: set[str],
    allow_missing_fields: bool,
    allow_unknown_fields: bool,
    unknown_fields_hint: str,
) -> None:
    """Raise or warn if the fields stored in the HDF5 file (`loaded_field_names`)
    do not match the fields in the class definition (`class_field_names`)."""
    missing_fields = loaded_field_names - class_field_names
    if missing_fields:
        msg = f"The following fields for {qualified_name!r} are stored in " \
              "the HDF5 file but are missing in the class definition: " \
              f"{', '.join(map(repr, sorted(missing_fields)))}."
        if allow_missing_fields:
            warnings.warn(msg, JaxonFormatWarning)
        else:
            raise JaxonFormatError(
                msg + " To omit this error run with allow_missing_fields=True.")
    unknown_fields = class_field_names - loaded_field_names
    if unknown_fields:
        msg = f"The following fields of {qualified_name!r} are present in " \
              "the class definition but are not stored in the HDF5 file: " \
              f"{', '.join(map(repr, sorted(unknown_fields)))}."
        if allow_unknown_fields:
            warnings.warn(msg + " " + unknown_fields_hint, JaxonFormatWarning)
        else:
            raise JaxonFormatError(
                msg \
                + " To omit this error run with allow_unknown_fields=True. " \
                + unknown_fields_hint)


def supply_dataclass_unmarshaler(
    allow_missing_fields: bool, allow_unknown_fields: bool
) -> Unmarshaler:
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
        _check_fields(
            get_qualified_name(instance),
            set(field_names),
            set(available_field_names),
            allow_missing_fields,
            allow_unknown_fields,
            "The fields will be initialized using their default_factory or default value. "
            "If both are missing JaxonNotLoaded() will be used as a placeholder. "
            "Note: __post_init__() is never called.",
        )
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


def supply_namedtuple_unmarshaler(
    allow_missing_fields: bool, allow_unknown_fields: bool
) -> Unmarshaler:
    """Create an unmarshaler for named tuples."""

    def namedtuple_unmarshaler(type_info: str, dict_: PyTree) -> PyTree | None:
        try:
            th, qualified_name = split_type_hint_and_type_arg(type_info)
        except JaxonFormatError:
            return None
        if th != "namedtuple":
            return None
        if qualified_name is None:
            raise JaxonFormatError("expected type argument for named tuple")
        if type(dict_) is not dict:
            raise JaxonFormatError("expected dict for marshaled named tuple")
        cls_ = get_class(qualified_name)
        if not is_namedtuple(cls_):
            raise JaxonFormatError(f"expected that {qualified_name!r} is a named tuple")
        field_names = set(cls_._fields)
        available_field_names = set(dict_.keys())
        _check_fields(
            qualified_name,
            field_names,
            available_field_names,
            allow_missing_fields,
            allow_unknown_fields,
            "The fields will be initialized using their default value. If there is "
            "none, JaxonNotLoaded() will be used as a placeholder.",)
        defaults = cls_._field_defaults
        values = []
        for name in cls_._fields:
            try:
                val = dict_[name]
            except KeyError:
                val = defaults.get(name, JaxonNotLoaded())
            values.append(val)
        # _make bypasses a custom __new__, like the dataclass unmarshaler bypasses __init__
        return cls_._make(values)

    return namedtuple_unmarshaler


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
    if not hasattr(cls_, "from_jaxon"):
        raise JaxonFormatError(
            f"expected that the class {qualified_name!r} marshaled via the "
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
            raise JaxonFormatError(f"unexpected type argument {ta!r} for type hint 'dill'")
        if not allow_dill:
            raise JaxonFormatError(
                "cannot load dill serialized object as allow_dill=False")
        return dill.loads(marshaled, **dill_kwargs)
    return dill_unmarshaler
