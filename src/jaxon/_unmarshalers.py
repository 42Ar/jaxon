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
    get_class, get_qualified_name, split_type_hint_and_type_arg
)


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
        missing_fields = available_field_names - field_names
        if missing_fields:
            message = (
                f"The following fields in {get_qualified_name(instance)!r} are present in "
                "the HDF5 file but are missing in the class "
                f"definition: {', '.join(map(repr, missing_fields))}"
            )
            if allow_missing_fields:
                warnings.warn(message, JaxonFormatWarning)
            else:
                raise JaxonFormatError(
                    message + ". To omit this error run with allow_missing_fields=True."
                )
        unknown_fields = field_names - available_field_names
        if unknown_fields:
            message = (
                f"the following fields in {get_qualified_name(instance)!r} are present in "
                "the class definition but are missing in the HDF5 "
                f"file: {', '.join(map(repr, unknown_fields))}"
            )
            if allow_unknown_fields:
                warnings.warn(message, JaxonFormatWarning)
            else:
                raise JaxonFormatError(
                    message + ". To omit this error run with "
                    "allow_unknown_fields=True. Missing fields will be initialized using the "
                    "default_factory or default value. If both are missing JaxonNotLoaded() "
                    "will be used as a placeholder. The __post_init__() logic is never triggered."
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
            "to_jaxon method has the from_jaxon method"
        )
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
            raise JaxonFormatError(
                "cannot load dill serialized object as allow_dill=False"
            )
        return dill.loads(marshaled, **dill_kwargs)

    return dill_unmarshaler
