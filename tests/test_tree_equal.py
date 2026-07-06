"""
constants.py

Meta tests for the `assert_tree_equal` method.

Author
------
Frank Hermann
"""

import pytest
import numpy as np
from .testing.tree_equal import assert_tree_equal


def test_tree_equal_error_if_titles_differ():
    x = np.array([(b'Rex', 9, 81.0), (b'Fido', 3, 27.0)],
        dtype=[('name', 'V10'), ('age', 'i4'), ('weight', 'f4')])
    y = np.array([(b'Rex', 9, 81.0), (b'Fido', 3, 27.0)],
        dtype=[('nameX', 'V10'), ('age', 'i4'), ('weight', 'f4')])
    with pytest.raises(AssertionError):
        assert_tree_equal(x, y)


def test_tree_equal_error_if_references_not_recovered():
    a1 = np.array([3, 4])
    a2 = np.array([3, 4])
    saved = {"a": a1, "b": a1}
    loaded = {"a": a1, "b": a2}
    with pytest.raises(AssertionError):
        assert_tree_equal(saved, loaded)
