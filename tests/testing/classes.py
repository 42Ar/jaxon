"""
classes.py

Defines custom classes for testing purposes.

Author
------
Frank Hermann
"""


from typing import override, Any
from dataclasses import dataclass
from .tree_equal import PyTreeTestNode, DillSerializedTestObject


class ObjectForDill(DillSerializedTestObject):
    a: float = 0.5

    def __hash__(self) -> int:
        return hash(self.a)

    @override
    def __eq__(self, other) -> bool:
        return self.a == other.a


class CustomTypeReturnDict(PyTreeTestNode):
    def __init__(self, a):
        self.a = a

    def from_jaxon(self, jaxon):
        self.a = jaxon["a"]

    def to_jaxon(self):
        return {"a": self.a}

    @override
    def children(self) -> tuple:
        return (self.a,)


class CustomTypeReturnField(PyTreeTestNode):
    def __init__(self, obj):
        self.obj = obj

    def from_jaxon(self, jaxon):
        self.obj = jaxon

    def to_jaxon(self):
        return self.obj

    @override
    def children(self) -> tuple:
        return (self.obj,)

    def __hash__(self):
        return hash(self.obj)


@dataclass
class CustomDataclass(PyTreeTestNode):
    a: Any
    b: Any = 1

    def __hash__(self):
        return hash((self.a, self.b))

    @override
    def children(self) -> tuple:
        return (self.a, self.b)
