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
import importlib
import sys
import types
import pytest
from jaxon._common import is_namedtuple, get_qualified_name
from .testing.classes import function_for_type_saving


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


def test_get_qualified_name_of_importable_objects():
    assert get_qualified_name(int) == "builtins.int"
    assert get_qualified_name(function_for_type_saving) == (
        "tests.testing.classes.function_for_type_saving")


def test_get_qualified_name_rejects_objects_without_module():
    namespace = {}
    exec("def function():\n    pass\n", namespace)
    no_module_function = namespace["function"]
    assert no_module_function.__module__ is None
    assert get_qualified_name(no_module_function) is None

    class NoModuleClass:
        pass

    NoModuleClass.__module__ = ""
    NoModuleClass.__qualname__ = "NoModuleClass"
    assert get_qualified_name(NoModuleClass) is None


def test_get_qualified_name_rejects_main_module():
    class MainClass:
        pass

    MainClass.__module__ = "__main__"
    assert get_qualified_name(MainClass) is None


def test_get_qualified_name_rejects_dynamic_modules():
    module_name = "jaxon_test_dynamic_module"
    module = types.ModuleType(module_name)

    class DynamicClass:
        pass

    DynamicClass.__module__ = module_name
    DynamicClass.__qualname__ = "DynamicClass"
    module.DynamicClass = DynamicClass
    sys.modules[module_name] = module
    try:
        assert importlib.import_module(module_name) is module
        assert get_qualified_name(DynamicClass) is None
    finally:
        del sys.modules[module_name]
