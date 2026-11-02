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
Built in marshalers which convert custom objects into pytrees of supported types.
"""

from dataclasses import fields, is_dataclass
import dill
from ._common import Marshaler, PyTree, get_class, get_qualified_name, is_namedtuple_class


def class_can_be_found(qualified_name: str) -> bool:
    """Check if the class can be found under `qualified_name`, which is required to load it again.
    This is not the case for classes defined inside functions, for example."""
    try:
        get_class(qualified_name)
    except (ImportError, AttributeError):
        return False
    return True


def dataclass_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
    if not is_dataclass(pytree):
        return None
    th = get_qualified_name(pytree)
    if not class_can_be_found(th):
        return None  # pass to the next marshaler
    res = {field.name: getattr(pytree, field.name) for field in fields(pytree)}
    return f"dclass[{th}]", res


def namedtuple_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
    if not is_namedtuple_class(type(pytree)):
        return None
    th = get_qualified_name(pytree)
    if not class_can_be_found(th):
        return None  # pass to the next marshaler
    res = pytree._asdict()
    return f"namedtuple[{th}]", res


def jaxon_interface_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
    if hasattr(pytree, "to_jaxon"):
        return f"jaxon[{get_qualified_name(pytree)}]", pytree.to_jaxon()
    return None


def supply_dill_marshaler(dill_kwargs) -> Marshaler:
    def dill_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
        return "dill", dill.dumps(pytree, **dill_kwargs)

    return dill_marshaler
