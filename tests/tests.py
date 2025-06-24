"""
test_util.py

Contains tests for the core functionality implemented in jaxon.__init__.py

Author
------
Frank Hermann
"""


from typing import Any
import tempfile
import random
import string
import unittest
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
import h5py
from .test_util import tree_equal
from jaxon import load, save, CircularPytreeException, JAXON_NP_NUMERIC_TYPES
from jaxon import JaxonStorageHints, JAXON_ROOT_GROUP_KEY


class TestObjectForDill:
    a = 0.5

    def __eq__(self, other):
        return self.a == other.a


class CustomTypeReturnDict:
    def __init__(self, a):
        self.a = a

    def from_jaxon(self, jaxon):
        self.a = jaxon["a"]

    def to_jaxon(self):
        return {"a": self.a}


class CustomTypeReturnCustom:
    def __init__(self, obj):
        self.obj = obj

    def from_jaxon(self, jaxon):
        self.obj = jaxon

    def to_jaxon(self):
        return self.obj

    def __eq__(self, other):
        return self.obj == other

    def __hash__(self):
        return hash(self.obj)


@dataclass
class CustomDataclass:
    mandatory: Any
    optional: Any = 345774

    def __hash__(self):
        return hash((self.mandatory, self.optional))


class RoundtripTests(unittest.TestCase):
    def do_roundtrip(self, pytree, exact_python_numeric_types, allow_dill=False,
                     downcast_to_base_types=None):
        with tempfile.TemporaryFile() as fp:
            save(fp, pytree, exact_python_numeric_types=exact_python_numeric_types,
                 downcast_to_base_types=downcast_to_base_types, allow_dill=allow_dill)
            return load(fp, allow_dill=allow_dill)

    def run_roundtrip_test(self, pytree, exact_python_numeric_types, allow_dill=False):
        loaded = self.do_roundtrip(pytree, exact_python_numeric_types, allow_dill)
        self.assertTrue(tree_equal(loaded, pytree, typematch=exact_python_numeric_types))

    def rand_string(self, seed, n):
        random.seed(seed)
        special = ["'", '"', "\0", "\r", "\n", "ä", "ö", "ü", "ß", ":", "\\"]
        return "".join(random.choices(list(string.ascii_uppercase) + special, k=n))

    def test_simple_types(self):
        pytree = {
            "complex": 1j + 5,
            "bool": True,
            "None": None,
            "string": "string",
            "string_with_qoutation1": "'",
            "string_with_qoutation2": '"',
            "string_with_qoutation3": '"\'',
            "string_with_zeros": '\0sfddf\0asdf',
            "string_with_trailing_zeros": '\0sfddf\0asdf\0\0',
            "string_with_trailing_zeros_and_non_ascii": '\0sfddf\0asdöüüäöüöäöüöüf\0\0'*5,
            "string_with_colons_1": ":sdffds:asd:::ads:",
            "string_with_colons_2": ":",
            ":": "234",
            "sdf:sdffds": "34",
            "'": "",
            '"': "",
            "\0sfddf\0asdf": "",
            "\0sfddf\0asdf\0\0": "",
            "\0sfddf\0asdöüüäöüöäöüöüf\0\0": "",
            "öäööääööäöä": "",
            "list": [4, "asf"],
            "tuple": (4, 3, "dsf", 5.5),
            "bytes": b"xfg",
            "bytes_with_zeros": b"sdf\0sdf\0\0sdf",
            "bytes_with_trailing_zeros": b"sdf\0sdf\0\0sdf\0\0",
            "int64": np.int64(313245),
            "float64": np.float64(3465.34),
            "int32": np.int32(487),
            "scalars": [scalar_type(0) for scalar_type in JAXON_NP_NUMERIC_TYPES],
            "npbool": np.bool(3465.34),
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
        for exact_python_numeric_types in (False, True):
            self.run_roundtrip_test(pytree, exact_python_numeric_types)

    def test_ararys(self):
        pytree = {
            "int32": np.arange(100, dtype=np.int32),
            "int64": np.arange(100, dtype=np.int64),
            "other": [np.zeros(100, dtype=scalar_type) for scalar_type in JAXON_NP_NUMERIC_TYPES],
            "jax": jnp.zeros((23, 21)),
            "jax2": jnp.array((23, 21))

        }
        for exact_python_numeric_types in (False, True):
            self.run_roundtrip_test(pytree, exact_python_numeric_types)

    def test_trivial_roots(self):
        for exact_python_numeric_types in (False, True):
            self.run_roundtrip_test(1, exact_python_numeric_types)
            self.run_roundtrip_test(None, exact_python_numeric_types)
            self.run_roundtrip_test({}, exact_python_numeric_types)
            self.run_roundtrip_test({"a": 345}, exact_python_numeric_types)
            self.run_roundtrip_test([], exact_python_numeric_types)
            self.run_roundtrip_test([3], exact_python_numeric_types)
            self.run_roundtrip_test(b"dfuikfhk\0\0ufs", exact_python_numeric_types)

    def test_dill_object_at_root(self):
        r = self.do_roundtrip(TestObjectForDill(), False, allow_dill=True)
        self.assertEqual(type(r), TestObjectForDill)
        self.assertEqual(r.a, 0.5)

    def test_dill_objects_in_container(self):
        pytree = [{"adssd": TestObjectForDill()}, TestObjectForDill()]
        for exact_python_numeric_types in (False, True):
            self.run_roundtrip_test(pytree, exact_python_numeric_types, allow_dill=True)

    def test_numeric_type_conversion(self):
        pytree = {"int": 3, "float": 45.4, "complex": 4j + 4, "bool": True}
        out = self.do_roundtrip(pytree, exact_python_numeric_types=False)
        self.assertEqual(type(out["int"]), np.int64)
        self.assertEqual(type(out["float"]), np.float64)
        self.assertEqual(type(out["complex"]), np.complex128)
        self.assertEqual(type(out["bool"]), np.bool)

    def test_type_downcast(self):
        class testint(int):
            pass
        class testint64(np.int64):
            pass
        pytree = {"testint": testint(), "testint64": testint64()}
        out = self.do_roundtrip(pytree, exact_python_numeric_types=True,
                                downcast_to_base_types=(testint, testint64))
        self.assertEqual(type(out["testint"]), int)
        self.assertEqual(type(out["testint64"]), np.int64)

    def test_container_type_downcast(self):
        class mydict(dict):
            pass
        class mylist(list):
            pass
        class mytuple(tuple):
            pass
        pytree = mydict({"mylist": mylist([12, 231, mylist(["ads"])]),
                         "mytuple": mytuple((324, 234, "df"))})
        out = self.do_roundtrip(pytree, exact_python_numeric_types=True,
                                downcast_to_base_types=[mydict, mylist, mytuple])
        self.assertEqual(type(out), dict)
        self.assertEqual(type(out["mylist"]), list)
        self.assertEqual(type(out["mytuple"]), tuple)

    def test_numeric_and_type_downcast(self):
        class testint(int):
            pass
        class testint64(np.int64):
            pass
        pytree = {"testint": testint(), "testint64": testint64()}
        out = self.do_roundtrip(pytree, exact_python_numeric_types=False,
                                downcast_to_base_types=(testint, testint64))
        self.assertEqual(type(out["testint"]), np.int64)
        self.assertEqual(type(out["testint64"]), np.int64)

    def test_custom_types(self):
        pytree = {
            "return_dict": CustomTypeReturnDict(3),
            "return_custom": CustomTypeReturnCustom(CustomTypeReturnDict(6)),
        }
        pytree = self.do_roundtrip(pytree, exact_python_numeric_types=True)
        self.assertEqual(type(pytree["return_dict"]), CustomTypeReturnDict)
        self.assertEqual(pytree["return_dict"].a, 3)
        self.assertEqual(type(pytree["return_custom"]), CustomTypeReturnCustom)
        self.assertEqual(type(pytree["return_custom"].obj), CustomTypeReturnDict)

    def test_single_big_attr_value(self):
        pytree = self.rand_string(42, 1000000)
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True)

    def test_multi_big_attr_value(self):
        pytree = [self.rand_string(i, 100000) for i in range(10)]
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True)

    def test_nonstring_dict_keys(self):
        pytree = {
            0: "ksdnkf",
            1: "asd",
            234: 5,
            (34, 234): 8,
            "sfddf": "dfs",

            # the reason why this works out of the box
            # is because the return value of jaxon type
            # can never be a simple atom (because it is a container)
            # and always must create a group
            CustomTypeReturnCustom((324, 34)): 24,
            CustomDataclass(234, "sdf"): "oasfd"
        }
        r = self.do_roundtrip(pytree, True)
        self.assertEqual(pytree, r)

    def test_single_big_key_value(self):
        pytree = {self.rand_string(42, 1000000), "val"}
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True)

    def test_multi_big_key_value(self):
        pytree = {self.rand_string(i, 100000): i for i in range(10)}
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True)

    def test_custom_dataclass(self):
        pytree = {CustomDataclass(213): CustomDataclass(CustomDataclass(21), "jkk")}
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True)


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


class IntrospectiveTests(unittest.TestCase):
    def test_store_in_dataclass(self):
        pytree = {"attribute": np.zeros(10), "dataset": np.zeros(10)}
        with tempfile.TemporaryFile() as fp:
            save(fp, pytree, storage_hints=[(pytree["dataset"], JaxonStorageHints(True))])
            with h5py.File(fp, 'r') as file:
                self.assertIn("'dataset'", list(file[JAXON_ROOT_GROUP_KEY]))
                self.assertEqual(1, len(list(file[JAXON_ROOT_GROUP_KEY])))
                self.assertNotIn("'attribute'", list(file[JAXON_ROOT_GROUP_KEY]))
