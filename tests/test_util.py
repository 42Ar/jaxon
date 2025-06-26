"""
test_util.py

Provides tree_equal which is only used during testing.

Author
------
Frank Hermann
"""


import jax.numpy as jnp
import numpy as np


def leaves_equal(t1, t2, strict_leaf_type_check=False):
    assert not strict_leaf_type_check or type(t1) is type(t2)
    if isinstance(t1, np.ndarray):
        assert isinstance(t2, np.ndarray)
        assert t1.dtype == t2.dtype
        assert np.array_equal(t1, t2, equal_nan=True)
    elif isinstance(t1, jnp.ndarray):
        assert isinstance(t2, jnp.ndarray)
        assert t1.dtype == t2.dtype
        assert np.array_equal(t1, t2, equal_nan=True)
    else:
        assert t1 == t2


def tree_equal(t1, t2, strict_leaf_type_check=False):
    if isinstance(t1, dict):
        assert type(t1) is type(t2)
        assert len(t1) == len(t2)
        for (k1, v1), (k2, v2) in zip(t1.items(), t2.items()):
            tree_equal(k1, k2)
            tree_equal(v1, v2)
    elif isinstance(t1, (list, tuple)):
        assert type(t1) is type(t2)
        assert len(t1) == len(t2)
        for v1, v2 in zip(t1, t2):
            tree_equal(v1, v2)
    elif isinstance(t1, (set, frozenset)):
        assert type(t1) is type(t2)
        assert len(t1) == len(t2)
        unmatched = list(t2)
        for x in t1:
            for y in unmatched:
                try:
                    tree_equal(x, y)
                except AssertionError:
                    pass
                break
            else:
                assert False
            unmatched.remove(y)
    else:
        leaves_equal(t1, t2, strict_leaf_type_check)
