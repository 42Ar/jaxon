"""
test_util.py

Provides tree_equal which is only used during testing.

Author
------
Frank Hermann
"""


import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from jaxon import JAXON_NP_NUMERIC_TYPES, JAXON_PY_NUMERIC_TYPES


class JaxonPyTreeTestNode(ABC):
    @abstractmethod
    def children(self) -> tuple:
        """Return all children in a tuple."""


def _is_type(obj, types, downcast_to_base_types):
    """Check if the type of `obj` is in `types` or if the user allowed downcasting to any
    of the types (if downcasting is possible)."""
    for t in types:
        if type(obj) is t or (type(obj) in downcast_to_base_types and isinstance(obj, t)):
            return True
    return False


def _is_non_reference_type(pytree, downcast_to_base_types) -> bool:
    return pytree is None or pytree is ... \
        or  _is_type(pytree, JAXON_NP_NUMERIC_TYPES, downcast_to_base_types) \
        or _is_type(pytree, JAXON_PY_NUMERIC_TYPES, downcast_to_base_types) \
        or isinstance(pytree, (range, slice, str))


def tree_equal(t1, t2, downcast_to_base_types=tuple(), py_to_np_types=tuple(), checked_objects_t1=None, checked_objects_t2=None):
    # check identity
    if t1 is t2:
        # under certain circumstances, the trees can share singleton
        # constants (specifically JAXON_NOT_LOADED); in other cases this
        # "is" check can shortcut checking (e.g. None is None, ...)
        return
    
    # check type compatibility
    allowed_base_types_t1 = {t for t in downcast_to_base_types if isinstance(t1, t)}
    if allowed_base_types_t1:
        # downcasting to at least one common base type was allowed
        assert any(isinstance(t2, base_type) for base_type in allowed_base_types_t1)
    elif isinstance(t1, py_to_np_types) or isinstance(t2, py_to_np_types):
        # one of them has been converted from a python to a numpy type
        # ignore types; content must match (this is not as strict as it could be)
        pass
    else:
        # normally, types must be exactly the same
        assert type(t1) is type(t2)
    
    # check if reference in both trees are the same
    if checked_objects_t1 is None or checked_objects_t2 is None:
        # this should happen only for the root
        assert checked_objects_t1 is None and checked_objects_t2 is None
        checked_objects_t1 = set()
        checked_objects_t2 = set()
    if not _is_non_reference_type(t1, downcast_to_base_types):
        # for non reference types, the references must be correct
        checked_t1 = id(t1) in checked_objects_t1
        checked_t2 = id(t2) in checked_objects_t2
        # if object has been (not) seen in t1 it must also have (not) been seen in t2
        # if this is not the case the references were not recovered correctly
        if checked_t1 != checked_t2:
            print(t1, t2)
        assert checked_t1 == checked_t2, "reference not correctly recovered"
        if checked_t1:
            return  # has already been checked
        checked_objects_t1.add(id(t1))
        checked_objects_t2.add(id(t2))
    
    # check if contents are the same
    if isinstance(t1, dict):
        assert type(t1) is type(t2)
        assert len(t1) == len(t2)
        for (k1, v1), (k2, v2) in zip(t1.items(), t2.items()):
            tree_equal(k1, k2, downcast_to_base_types, py_to_np_types, checked_objects_t1, checked_objects_t2)
            tree_equal(v1, v2, downcast_to_base_types, py_to_np_types, checked_objects_t1, checked_objects_t2)
    elif isinstance(t1, (list, tuple)):
        assert type(t1) is type(t2)
        assert len(t1) == len(t2)
        for v1, v2 in zip(t1, t2):
            tree_equal(v1, v2, downcast_to_base_types, py_to_np_types, checked_objects_t1, checked_objects_t2)
    elif isinstance(t1, (set, frozenset)):
        assert type(t1) is type(t2)
        assert len(t1) == len(t2)
        unmatched = list(t2)
        for x in t1:
            for y in unmatched:
                try:
                    tree_equal(x, y, downcast_to_base_types, py_to_np_types, checked_objects_t1, checked_objects_t2)
                except AssertionError:
                    continue
                break
            else:
                assert False, f"frozenset/set element {x!r} exists in t1 but not found in t2"
            unmatched.remove(y)
    elif isinstance(t1, JaxonPyTreeTestNode):
        assert isinstance(t2, JaxonPyTreeTestNode)
        t1_children = t1.children()
        t2_children = t2.children()
        assert len(t1_children) == len(t2_children)
        for a, b in zip(t1_children, t2_children):
            tree_equal(a, b, downcast_to_base_types, py_to_np_types, checked_objects_t1, checked_objects_t2)
    elif isinstance(t1, np.ndarray):
        assert isinstance(t2, np.ndarray)
        assert t1.dtype == t2.dtype
        assert np.array_equal(t1, t2, equal_nan=True)
    elif isinstance(t1, jnp.ndarray):
        assert isinstance(t2, jnp.ndarray)
        assert t1.dtype == t2.dtype
        assert np.array_equal(t1, t2, equal_nan=True)
    else:
        assert t1 == t2
