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
from ._common import Marshaler, PyTree, get_qualified_name, is_namedtuple


def dataclass_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
    if not is_dataclass(pytree):
        return None
    th = get_qualified_name(type(pytree))
    if th is None:
        return None  # pass to the next marshaler
    res = {field.name: getattr(pytree, field.name) for field in fields(pytree)}
    return f"dclass[{th}]", res


def namedtuple_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
    if not is_namedtuple(type(pytree)):
        return None
    th = get_qualified_name(type(pytree))
    if th is None:
        return None  # pass to the next marshaler
    res = pytree._asdict()
    return f"namedtuple[{th}]", res


def jaxon_interface_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
    if hasattr(pytree, "to_jaxon"):
        th = get_qualified_name(type(pytree))
        if th is None:
            return None  # pass to the next marshaler
        return f"jaxon[{th}]", pytree.to_jaxon()
    return None


def supply_dill_marshaler(dill_kwargs) -> Marshaler:
    def dill_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
        return "dill", dill.dumps(pytree, **dill_kwargs)

    return dill_marshaler
