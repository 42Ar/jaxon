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
