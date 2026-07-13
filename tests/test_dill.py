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
Test dill using the fuzzer and `assert_tree_equal`. This is to establish the baseline
that objects serialized with dill cannot cause any failed tests.

Author
------
Frank Hermann
"""

import dill
from .testing.fuzz import fuzz_tree_generator
from .testing.tree_equal import assert_tree_equal


def test_fuzzing_dill():
    for pytree in fuzz_tree_generator(1000):
        loaded = dill.loads(dill.dumps(pytree))
        assert_tree_equal(pytree, loaded)
