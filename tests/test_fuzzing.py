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
Meta tests for the fuzzer.

Author
------
Frank Hermann
"""


from numpy.random import default_rng
from .testing.fuzz import rand_str, rand_bytes, FUZZ_CHARS, FUZZ_BYTES


def test_rand_str():
    assert rand_str(default_rng(0), 0) == ""
    s = rand_str(default_rng(0), 100)
    print(s)
    assert type(s) is str
    assert len(s) == 100
    assert all(c in FUZZ_CHARS for c in s)


def test_rand_bytes():
    assert rand_bytes(default_rng(0), 0) == b""
    s = rand_bytes(default_rng(0), 100)
    print(s)
    assert type(s) is bytes
    assert len(s) == 100
    assert all(bytes([c]) in FUZZ_BYTES for c in s)
