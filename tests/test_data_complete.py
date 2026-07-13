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
test_data_complete.py

Meta tests for checking if the data provided by `testing.data` is complete
and in the expected format.

Author
------
Frank Hermann
"""


from jaxon import JAXON_ATOMIC_TYPES, JAXON_NUMPY_ARRAY_TYPES
from .testing.data import TEST_JAXON_ATOMIC, TEST_NUMPY_ARRAY_VALUES


def test_tests_sets_complete():
    assert JAXON_ATOMIC_TYPES == set(TEST_JAXON_ATOMIC.keys())
    assert JAXON_NUMPY_ARRAY_TYPES == set(TEST_NUMPY_ARRAY_VALUES.keys())


def test_tests_sets_consistent():
    for test_set in (TEST_JAXON_ATOMIC, TEST_NUMPY_ARRAY_VALUES):
        for tp, examples in test_set.items():
            assert type(examples) is tuple
            for x in examples:
                assert tp is type(x)
