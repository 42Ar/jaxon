"""
test_util.py

Provides tree_equal which is used during testing.
Some of the functions are adapted copies from the
equinox package by Patrick Kidger, which is MIT licensed.

Author
------
Frank Hermann
"""


import jax.tree_util as jtu
import jax
import jax.numpy as jnp
import numpy as np


def is_array(element):
    """Returns `True` if `element` is a JAX array or NumPy array."""
    return isinstance(element, (np.ndarray, np.generic, jax.Array))


def _is_nonscalar_array(a):
    return is_array(a) and a.ndim > 0


def _array_equal(x, y, npi, rtol, atol):
    assert x.dtype == y.dtype
    if (
        isinstance(rtol, (int, float))
        and isinstance(atol, (int, float))
        and rtol == 0
        and atol == 0
    ) or not npi.issubdtype(x.dtype, npi.inexact):
        return npi.all(x == y)
    else:
        return npi.allclose(x, y, rtol=rtol, atol=atol)


def tree_equal(*pytrees, typematch = False, rtol = 0.0, atol = 0.0):
    """Almost identical to equinox.tree_util.tree_equal but fixes a bug when comparing a
    zero dimensional array to a python scalar with typematch = False."""
    flat, treedef = jtu.tree_flatten(pytrees[0])
    traced_out = True
    for pytree in pytrees[1:]:
        flat_, treedef_ = jtu.tree_flatten(pytree)
        if treedef_ != treedef:
            return False
        assert len(flat) == len(flat_)
        for elem, elem_ in zip(flat, flat_):
            if typematch:
                if type(elem) != type(elem_):
                    return False
            if isinstance(elem, (np.ndarray, np.generic)) and isinstance(
                elem_, (np.ndarray, np.generic)
            ):
                if (
                    (elem.shape != elem_.shape)
                    or (elem.dtype != elem_.dtype)
                    or not _array_equal(elem, elem_, np, rtol, atol)
                ):
                    return False
            elif _is_nonscalar_array(elem):
                if _is_nonscalar_array(elem_):
                    if (elem.shape != elem_.shape) or (elem.dtype != elem_.dtype):
                        return False
                    traced_out = traced_out & _array_equal(elem, elem_, jnp, rtol, atol)
                else:
                    return False
            else:
                if _is_nonscalar_array(elem_):
                    return False
                else:
                    if elem != elem_:
                        return False
    return traced_out
