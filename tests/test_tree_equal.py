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


"""
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
