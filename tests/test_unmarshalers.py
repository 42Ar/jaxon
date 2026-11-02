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


"""Tests for the built-in unmarshalers."""

import pytest
from jaxon import JaxonFormatError
from jaxon._unmarshalers import supply_dataclass_unmarshaler, supply_namedtuple_unmarshaler, \
    jaxon_interface_unmarshaler, supply_dill_unmarshaler
from .testing.classes import CustomDataclass, CustomNamedTuple


def test_unmarshalers_ignore_other_or_invalid_type_hints():
    """Unmarshalers should ignore other or invalid type hints by returning 
    None (passing it to the next unmarshaler in the chain)."""
    for unmarshaler in (supply_dataclass_unmarshaler(False, False),
                        supply_namedtuple_unmarshaler(False, False),
                        jaxon_interface_unmarshaler,
                        supply_dill_unmarshaler(True, {})):
        # invalid type hint syntax: 
        for type_info in (
            "_OTHER[a.B]", "_OTHER[a[b]]", 
            "_OTHER[a[b]", "_OTHER]", "_OTHER[a" # invalid type hints (multiple or unmatched '['/']').
        ):
            assert unmarshaler(type_info, {}) is None


def test_unmarshalers_raise_on_invalid_type_hints_and_data():
    """Raise a JaxonFormatError when the type hint or data is invalid for given unmarshaler."""
    dataclass_name = f"{CustomDataclass.__module__}.CustomDataclass"
    namedtuple_name = f"{CustomNamedTuple.__module__}.CustomNamedTuple"
    dataclass_unmarshaler = supply_dataclass_unmarshaler(False, False)
    namedtuple_unmarshaler = supply_namedtuple_unmarshaler(False, False)
    dill_unmarshaler = supply_dill_unmarshaler(True, {})
    marshalers = [
        (dataclass_unmarshaler, "dclass", {}, "expected type argument"),
        (dataclass_unmarshaler, f"dclass[{dataclass_name}]", [1], "expected dict"),
        (dataclass_unmarshaler, f"dclass[{namedtuple_name}]", {}, "is a dataclass"),
        (namedtuple_unmarshaler, "namedtuple", {}, "expected type argument"),
        (namedtuple_unmarshaler, f"namedtuple[{namedtuple_name}]", [1], "expected dict"),
        (namedtuple_unmarshaler, f"namedtuple[{dataclass_name}]", {}, "is a named tuple"),
        (jaxon_interface_unmarshaler, "jaxon", {}, "expected type argument"),
        (jaxon_interface_unmarshaler, f"jaxon[{dataclass_name}]", {}, "from_jaxon"),
        (dill_unmarshaler, "dill[x]", "x", "unexpected type argument"),
    ]
    for unmarshaler, type_info, marshaled, message in marshalers:
        with pytest.raises(JaxonFormatError, match=message):
            unmarshaler(type_info, marshaled)
