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


"""Tests for helpers in `jaxon._common`."""

from collections import namedtuple
from typing import NamedTuple
import pytest
from jaxon._common import is_namedtuple


CollectionNamedTuple = namedtuple("CollectionNamedTuple", ["a", "b"])


class TypingNamedTuple(NamedTuple):
    a: int
    b: int = 1


class NamedTupleSubclass(CollectionNamedTuple):
    pass


class FakeNamedTuple:
    _fields = ("a", "b")
    _make = staticmethod(lambda values: values)
    _asdict = lambda self: {}
    _replace = lambda self, **kwargs: self
    _field_defaults = {}


class FakeTupleSubclass(tuple):
    _fields = ("a", "b")


class TupleSubclassWithInvalidFields(tuple):
    _fields = "a b"
    _make = staticmethod(lambda values: values)
    _asdict = lambda self: {}
    _replace = lambda self, **kwargs: self
    _field_defaults = {}


@pytest.mark.parametrize("cls_", [CollectionNamedTuple, TypingNamedTuple])
def test_is_namedtuple_true(cls_):
    assert is_namedtuple(cls_)


@pytest.mark.parametrize("cls_", [
    tuple,
    CollectionNamedTuple(1, 2),
    NamedTupleSubclass,
    FakeNamedTuple,
    FakeTupleSubclass,
    TupleSubclassWithInvalidFields,
])
def test_is_namedtuple_false(cls_):
    assert not is_namedtuple(cls_)
