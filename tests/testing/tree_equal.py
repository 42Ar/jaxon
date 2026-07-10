"""
tree_equal.py

Provides `assert_tree_equal` which is used during testing.

Author
------
Frank Hermann
"""

from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
from jaxon._common import _JAXON_JAX_ARRAY_TYPE, JAXON_PY_NUMERIC_TYPES, \
    JAXON_NUMPY_ATOMIC_TYPES, JaxonNotLoaded, JaxonNumpyNumeric, JAXON_PY_CONTAINER_TYPES
from jaxon._save import _base_type


class PyTreeTestNode(ABC):
    @abstractmethod
    def children(self) -> tuple:
        """Return all children in a tuple."""


class DillObject:
    """Flag to indicate an object that is serialized with dill."""


def _is_non_reference_type(pytree, downcast_to_base_types, py_to_np_types) -> bool:
    """If this function returns True it doesn't mean that the type cannot be
    referenced within the hdf5 file, it just means that a reference check should
    not be performed because the type might be interned by python."""
    base_type = _base_type(pytree, JAXON_PY_NUMERIC_TYPES | {str, bytes, tuple, frozenset},
        downcast_to_base_types)
    return pytree is None or pytree is Ellipsis \
        or (base_type is not None and base_type not in py_to_np_types) \
        or isinstance(pytree, range)


def _assert_real_equal(x, y):
    assert x == y or (np.isnan(x) and np.isnan(y))


def _assert_complex_equal(a, b):
    _assert_real_equal(a.real, b.real)
    _assert_real_equal(a.imag, b.imag)


def _assert_numbers_equal(saved, loaded):
    assert type(saved) is type(loaded)
    if np.iscomplex(saved):
        _assert_complex_equal(saved, loaded)
    else:
        _assert_real_equal(saved, loaded)


def _is_type(saved, loaded, types, downcast_to_base_types):
    base_type = _base_type(saved, types, downcast_to_base_types)
    assert base_type is None or base_type is type(loaded)
    return base_type is not None


def assert_array_equal(a, b, xp):
    assert a.dtype == b.dtype and a.shape == b.shape
    if a.dtype.names is not None:
        for name in a.dtype.names:
            assert_array_equal(a[name], b[name], xp)
    elif xp.issubdtype(a.dtype, xp.inexact):
        assert xp.array_equal(a, b, equal_nan=True)
    else:
        assert xp.array_equal(a, b)


def _assert_tree_equal(saved, loaded, downcast_to_base_types, py_to_np_types,
        checked_objects_saved, checked_objects_loaded):
    """Note: This function is not exactly symmetric under swapping of saved and loaded."""
    if type(saved) is JaxonNotLoaded:
        # it is assumed that the testing code placed JaxonNotLoaded in saved where it is expected
        assert type(loaded) is JaxonNotLoaded
        return
    assert type(saved) is not JaxonNotLoaded, "found JaxonNotLoaded where it should not be"

    # check if references in both trees are the same
    if not _is_non_reference_type(saved, downcast_to_base_types, py_to_np_types):
        checked_saved = id(saved) in checked_objects_saved
        checked_loaded = id(loaded) in checked_objects_loaded
        # if an object has (not) been seen in saved it must also have (not) been seen in loaded
        # if this is not the case the references were not recovered correctly
        assert checked_saved == checked_loaded, "reference not correctly recovered"
        if checked_saved:
            return  # has already been checked
        checked_objects_saved.add(id(saved))
        checked_objects_loaded.add(id(loaded))

    # check if types and contents are the same
    if saved is None or saved is Ellipsis:
        assert saved is loaded
        return
    py_numeric_type = _base_type(saved, JAXON_PY_NUMERIC_TYPES, downcast_to_base_types)
    if py_numeric_type is not None:
        if py_numeric_type in py_to_np_types:
            saved = np.array(saved)[()]
            assert type(saved) is type(loaded)
        else:
            assert py_numeric_type is type(loaded)
            saved = np.array(saved)[()]
            loaded = np.array(loaded)[()]
        _assert_numbers_equal(saved, loaded)
        return
    if type(saved) is range:
        assert type(loaded) is range
        assert saved == loaded
        return
    if _is_type(saved, loaded, JAXON_NUMPY_ATOMIC_TYPES, downcast_to_base_types):
        if isinstance(saved, JaxonNumpyNumeric):
            _assert_numbers_equal(saved, loaded)
        else:
            # trailing null bytes do not need to be preserved for np.bytes_ and np.str_
            # this is standard behavior
            if type(loaded) is np.bytes_:
                assert loaded.rstrip(b"\x00") == saved.rstrip(b"\x00")
            elif type(loaded) is np.str_:
                assert loaded.rstrip("\x00") == saved.rstrip("\x00")
            else:
                assert loaded == saved
        return
    if _is_type(saved, loaded, (str,), downcast_to_base_types):
        assert saved == loaded
        return
    if _is_type(saved, loaded, (np.ndarray,), downcast_to_base_types):
        assert_array_equal(saved, loaded, np)
        return
    if _is_type(saved, loaded, (_JAXON_JAX_ARRAY_TYPE,), downcast_to_base_types):
        assert_array_equal(saved, loaded, jnp)
        return
    if _is_type(saved, loaded, (bytes, bytearray), downcast_to_base_types):
        assert saved == loaded
        return
    if _is_type(saved, loaded, JAXON_PY_CONTAINER_TYPES, downcast_to_base_types):
        if type(loaded) == dict:
            assert len(saved) == len(loaded)
            for (k1, v1), (k2, v2) in zip(saved.items(), loaded.items()):
                _assert_tree_equal(k1, k2, downcast_to_base_types, py_to_np_types,
                                   checked_objects_saved, checked_objects_loaded)
                _assert_tree_equal(v1, v2, downcast_to_base_types, py_to_np_types,
                                   checked_objects_saved, checked_objects_loaded)
            return
        if type(loaded) in (list, tuple):
            assert len(saved) == len(loaded)
            for v1, v2 in zip(saved, loaded):
                _assert_tree_equal(v1, v2, downcast_to_base_types, py_to_np_types,
                                   checked_objects_saved, checked_objects_loaded)
            return
        if type(loaded) in (set, frozenset):
            assert len(saved) == len(loaded)
            unmatched = list(loaded)
            for x in saved:
                for i, y in enumerate(unmatched):
                    checked_objects_saved_cpy = set(checked_objects_saved)
                    checked_objects_loaded_cpy = set(checked_objects_loaded)
                    try:
                        _assert_tree_equal(x, y, downcast_to_base_types, py_to_np_types, checked_objects_saved_cpy, checked_objects_loaded_cpy)
                    except AssertionError:
                        continue
                    checked_objects_saved.clear()
                    checked_objects_saved |= checked_objects_saved_cpy
                    checked_objects_loaded.clear()
                    checked_objects_loaded |= checked_objects_loaded_cpy
                    break
                else:
                    assert False, f"frozenset/set element {x!r} exists in saved but not found in loaded"
                del unmatched[i]
            return
        assert False, f"unknown container type {type(saved)!r}"
    if isinstance(loaded, PyTreeTestNode):
        if isinstance(saved, DillObject):
            # special handling for descending into dill object
            # references of objects inside the dilled object to jaxon objects are broken
            # because they cannot be preserved
            downcast_to_base_types = tuple()
            py_to_np_types = tuple()
            checked_objects_saved = set()
            checked_objects_loaded = set()
        assert type(loaded) is type(saved)
        ch1 = loaded.children()
        ch2 = saved.children()
        assert len(ch1) == len(ch2)
        for c1, c2 in zip(ch1, ch2):
            _assert_tree_equal(c1, c2, downcast_to_base_types, py_to_np_types, checked_objects_saved, checked_objects_loaded)
        return
    assert False, f"unknown node or leaf type in saved tree {type(saved)!r}"


def assert_tree_equal(saved, loaded, downcast_to_base_types=tuple(), py_to_np_types=tuple()):
    _assert_tree_equal(saved, loaded, downcast_to_base_types, py_to_np_types, set(), set())
