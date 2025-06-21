"""
test_util.py

Contains tests that first save, then load and finally check the loaded pytree for equality
against the original.

Author
------
Frank Hermann
"""

import tempfile
import numpy as np
import unittest
from .test_util import tree_equal
from jaxon import load, save, JAXON_SCALAR_BINARY_TYPES


class RoundtripTests(unittest.TestCase):
    def run_roundtrip_test(self, pytree, exact_python_types):
        with tempfile.TemporaryFile() as fp:
            save(fp, pytree, exact_python_types=exact_python_types)
            loaded = load(fp)
        self.assertTrue(tree_equal(loaded, pytree, typematch=exact_python_types, rtol=0, atol=0))

    def test_simple_types(self):
        pytree = {
            "complex": 1j + 5,
            "bool": True,
            "None": None,
            "test_string": "string",
            "list": [4, "asf"],
            "tuple": (4, 3, "dsf", 5.5),
            "bytes": b"xfg",
            "int64": np.int64(313245),
            "float64": np.float64(3465.34),
            "int32": np.int32(487),
            "scalars": [scalar_type(0) for scalar_type in JAXON_SCALAR_BINARY_TYPES],
            "bool": np.bool(3465.34),
            "complex128": np.complex128(123 + 32j),
            "key/with/slashes": {
                "more/slahes": 5
            }
        }
        for exact_python_types in (False, True):
            self.run_roundtrip_test(pytree, exact_python_types)

    def test_numpy_ararys(self):
        pytree = {
            "int32": np.arange(100, dtype=np.int32),
            "int64": np.arange(100, dtype=np.int64),
            "other": [np.zeros(100, dtype=scalar_type) for scalar_type in JAXON_SCALAR_BINARY_TYPES],
        }
        for exact_python_types in (False, True):
            self.run_roundtrip_test(pytree, exact_python_types)

    def test_trivial_roots(self):
        for exact_python_types in (False, True):
            self.run_roundtrip_test(1, exact_python_types)
            self.run_roundtrip_test(None, exact_python_types)
            self.run_roundtrip_test({}, exact_python_types)
            self.run_roundtrip_test({"a": 345}, exact_python_types)
            self.run_roundtrip_test([], exact_python_types)
            self.run_roundtrip_test([3], exact_python_types)
            self.run_roundtrip_test(b"dfuikfhkufs", exact_python_types)
