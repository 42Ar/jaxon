"""
test_util.py

Contains tests for the core functionality implemented in jaxon.__init__.py

Author
------
Frank Hermann
"""

import tempfile
import numpy as np
import unittest
from .test_util import tree_equal
from jaxon import load, save, JAXON_NP_NUMERIC_TYPES, CircularPytreeException


class TestObjectForDill:
    a = 0.5

    def __eq__(self, other):
        return self.a == other.a


class RoundtripTests(unittest.TestCase):
    def do_roundtrip(self, pytree, exact_python_types, allow_dill=False):
        with tempfile.TemporaryFile() as fp:
            save(fp, pytree, exact_python_types=exact_python_types, allow_dill=allow_dill)
            return load(fp, allow_dill=allow_dill)

    def run_roundtrip_test(self, pytree, exact_python_types, allow_dill=False):
        loaded = self.do_roundtrip(pytree, exact_python_types, allow_dill)
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
            "bytes_with_zeros": b"sdf\0sdf\0\0sdf",
            "bytes_with_trailing_zeros": b"sdf\0sdf\0\0sdf\0\0",
            "int64": np.int64(313245),
            "float64": np.float64(3465.34),
            "int32": np.int32(487),
            "scalars": [scalar_type(0) for scalar_type in JAXON_NP_NUMERIC_TYPES],
            "bool": np.bool(3465.34),
            "complex128": np.complex128(123 + 32j),
            "key/with/slashes": {
                "more/slahes": 5
            },
            "set": {231, "afsdd", 2342, "weffd"},
            "fset": frozenset([234, 234, 234]),
            "range1": range(23),
            "range2": range(2, 23),
            "range3": range(2, 2000, 23),
            "ellipsis": ...,
            "bytearrray": bytearray(b"xcvx<cv\0\0"),
            "memoryview": memoryview(b"xcvx<cv\0\0"),
            "slice1": slice(2),
            "slice2": slice(2, 2143),
            "slice3": slice(2, 2132, 23)
        }
        for exact_python_types in (False, True):
            self.run_roundtrip_test(pytree, exact_python_types)

    def test_numpy_ararys(self):
        pytree = {
            "int32": np.arange(100, dtype=np.int32),
            "int64": np.arange(100, dtype=np.int64),
            "other": [np.zeros(100, dtype=scalar_type) for scalar_type in JAXON_NP_NUMERIC_TYPES],
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
            self.run_roundtrip_test(b"dfuikfhk\0\0ufs", exact_python_types)

    def test_nonstring_dict_keys(self):
        pytree = {
        }
        for exact_python_types in (False, True):
            self.run_roundtrip_test(pytree, exact_python_types)

    def test_dill_object_at_root(self):
        r = self.do_roundtrip(TestObjectForDill(), False, allow_dill=True)
        self.assertEqual(type(r), TestObjectForDill)
        self.assertEqual(r.a, 0.5)

    def test_dill_objects_in_container(self):
        pytree = [{"adssd": TestObjectForDill()}, TestObjectForDill()]
        for exact_python_types in (False, True):
            self.run_roundtrip_test(pytree, exact_python_types, allow_dill=True)


class ErrorBranchTests(unittest.TestCase):
    def trigger_circular_reference_exception(self):
        pytree = {}
        pytree["a"] = pytree
        with tempfile.TemporaryFile() as fp:
            save(fp, pytree)

    def test_circular_reference_detection(self):
        self.assertRaises(CircularPytreeException, self.trigger_circular_reference_exception)

    def trigger_unsupported_type_exception(self):
        with tempfile.TemporaryFile() as fp:
            class custom:
                pass
            save(fp, custom())

    def test_unsupported_object(self):
        self.assertRaises(TypeError, self.trigger_unsupported_type_exception)
