"""
classes.py

Defines custom classes for testing purposes.

Author
------
Frank Hermann
"""


from typing import override, Any
from dataclasses import dataclass
from .tree_equal import PyTreeTestNode, DillObject


class ObjectForDill(PyTreeTestNode, DillObject):
    def __init__(self, a):
        self.a = a

    def __hash__(self) -> int:
        return hash(self.a)

    def __repr__(self) -> str:
        return f"ObjectForDill({self.a!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, ObjectForDill) and self.a == other.a

    @override
    def children(self) -> tuple:
        return (self.a,)


class CustomTypeReturnDict(PyTreeTestNode):
    def __init__(self, a):
        self.a = a

    def from_jaxon(self, jaxon):
        self.a = jaxon["a"]

    def to_jaxon(self):
        return {"a": self.a}

    def __hash__(self) -> int:
        return hash(self.a)

    def __repr__(self) -> str:
        return f"CustomTypeReturnDict({self.a!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, CustomTypeReturnDict) and self.a == other.a

    @override
    def children(self) -> tuple:
        return (self.a,)


class CustomTypeReturnTuple(PyTreeTestNode):
    def __init__(self, a):
        self.a = a

    def from_jaxon(self, jaxon):
        self.a = jaxon[0]

    def to_jaxon(self):
        return (self.a,)

    def __hash__(self) -> int:
        return hash(self.a)

    def __repr__(self) -> str:
        return f"CustomTypeReturnField({self.a!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, CustomTypeReturnTuple) and self.a == other.a

    @override
    def children(self) -> tuple:
        return (self.a,)


@dataclass
class CustomDataclass(PyTreeTestNode):
    a: Any
    b: Any = 1

    def __hash__(self):
        return hash((self.a, self.b))

    @override
    def children(self) -> tuple:
        return (self.a, self.b)
