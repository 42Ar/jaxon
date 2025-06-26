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
from jaxon import load, save, CircularPytreeException, JAXON_NP_NUMERIC_TYPES
from jaxon import JaxonStorageHints, JAXON_ROOT_GROUP_KEY
from .test_util import tree_equal


TEST_TYPES = (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
              np.uint64, np.float16, np.float32, np.float64, np.bool)
TEST_TYPES_FOR_COMPELX = (np.float32, np.float64)


class TestObjectForDill:
    a = 0.5

    def __eq__(self, other):
        tree_equal(self.a, other.a)
        return True
    
    def __hash__(self) -> int:
        return hash(self.a)


class TestCustomTypeReturnDict:
    def __init__(self, a):
        self.a = a

    def from_jaxon(self, jaxon):
        self.a = jaxon["a"]

    def to_jaxon(self):
        return {"a": self.a}

    def __eq__(self, other):
        tree_equal(self.a, other.a)
        return True


class TestCustomTypeReturnField:
    def __init__(self, obj):
        self.obj = obj

    def from_jaxon(self, jaxon):
        self.obj = jaxon

    def to_jaxon(self):
        return self.obj

    def __eq__(self, other):
        tree_equal(self.obj, other.obj)  # raises error
        return True

    def __hash__(self):
        return hash(self.obj)


@dataclass
class TestCustomDataclass:
    mandatory: Any
    optional: Any = 345774

    def __hash__(self):
        return hash((self.mandatory, self.optional))

    def __eq__(self, other) -> bool:
        tree_equal(self.mandatory, other.mandatory)
        tree_equal(self.optional, other.optional)
        return True


def build_fuzz_tree(cur_depth, max_depth, only_hashable=False):
    if random.random() < 0.2:
        subtree = build_fuzz_tree(cur_depth, max_depth, only_hashable=only_hashable)
        return TestCustomDataclass(subtree)
    if random.random() < 0.5 and cur_depth < max_depth:
        if not only_hashable and random.random() < 0.1:
            container = random.choice((list,))
            return container([build_fuzz_tree(cur_depth + 1, max_depth, only_hashable=True)
                             for _ in range(random.randint(0, 5))])
        if not only_hashable and random.random() < 0.4:
            return {build_fuzz_tree(cur_depth + 1, max_depth, only_hashable=True):
                    build_fuzz_tree(cur_depth + 1, max_depth, only_hashable=False)
                    for _ in range(random.randint(0, 5))}
        return tuple(build_fuzz_tree(cur_depth + 1, max_depth, only_hashable)
                     for _ in range(random.randint(0, 5)))
    if random.random() < 0.5:
        if only_hashable:
            return 3984789438723
        return np.arange(3)
    return np.int16(2)


class RoundtripTests(unittest.TestCase):
    def do_roundtrip(self, pytree, exact_python_numeric_types, allow_dill=False,
                     downcast_to_base_types=None):
        with tempfile.TemporaryFile() as fp:
            save(fp, pytree, exact_python_numeric_types=exact_python_numeric_types,
                 downcast_to_base_types=downcast_to_base_types, allow_dill=allow_dill)
            return load(fp, allow_dill=allow_dill)

    def run_roundtrip_test(self, pytree, exact_python_numeric_types, allow_dill=False):
        loaded = self.do_roundtrip(pytree, exact_python_numeric_types, allow_dill)
        tree_equal(loaded, pytree, strict_leaf_type_check=exact_python_numeric_types)
        return loaded

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
        nprng = np.random.default_rng(42)
        def random_complex(scalar_type):
            real = nprng.uniform(size=(4, 2, 3)).astype(scalar_type)
            imag = nprng.uniform(size=(4, 2, 3)).astype(scalar_type)
            return real + 1j*imag
        pytree = {
            "normal": nprng.uniform(size=(4, 2, 3)),
            "int32": (nprng.uniform(size=(4, 2, 3))*10000).astype(np.int32),
            "int64": (nprng.uniform(size=(4, 2, 3))*100).astype(np.int64),
            "other": [(nprng.uniform(size=(4, 2, 3))*100).astype(scalar_type) for scalar_type in TEST_TYPES],
            "jax":  [jnp.array((nprng.uniform(size=(4, 2, 3))*100).astype(scalar_type)) for scalar_type in TEST_TYPES],
            "complex": [random_complex(scalar_type) for scalar_type in TEST_TYPES_FOR_COMPELX],
            "complex_jax": [jnp.array(random_complex(scalar_type)) for scalar_type in TEST_TYPES_FOR_COMPELX]
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
            self.run_roundtrip_test(np.arange(2), exact_python_numeric_types)
            self.run_roundtrip_test(jnp.arange(2), exact_python_numeric_types)

    def test_dill_object_at_root(self):
        self.run_roundtrip_test(TestObjectForDill(), False, allow_dill=True)

    def test_dill_objects_in_container(self):
        pytree = [{"adssd": TestObjectForDill()}, TestObjectForDill()]
        for exact_python_numeric_types in (False, True):
            self.run_roundtrip_test(pytree, exact_python_numeric_types, allow_dill=True)

    def test_numeric_type_conversion(self):
        pytree = {"int": 3, "float": 45.4, "complex": 4j + 4, "bool": True}
        out = self.run_roundtrip_test(pytree, exact_python_numeric_types=False)
        self.assertEqual(type(out["int"]), np.int64)
        self.assertEqual(type(out["float"]), np.float64)
        self.assertEqual(type(out["complex"]), np.complex128)
        self.assertEqual(type(out["bool"]), np.bool)

    def test_type_downcast(self):
        class TestInt(int):
            pass
        class TestInt64(np.int64):
            pass
        pytree = {"testint": TestInt(), "testint64": TestInt64()}
        out = self.do_roundtrip(pytree, exact_python_numeric_types=True,
                                downcast_to_base_types=(TestInt, TestInt64))
        self.assertEqual(type(out["testint"]), int)
        self.assertEqual(type(out["testint64"]), np.int64)

    def test_container_type_downcast(self):
        class TestDict(dict):
            pass
        class TestList(list):
            pass
        class TestTuple(tuple):
            pass
        pytree = TestDict({"mylist": TestList([12, 231, TestList(["ads"])]),
                         "mytuple": TestTuple((324, 234, "df"))})
        out = self.do_roundtrip(pytree, exact_python_numeric_types=True,
                                downcast_to_base_types=[TestDict, TestList, TestTuple])
        self.assertEqual(type(out), dict)
        self.assertEqual(type(out["mylist"]), list)
        self.assertEqual(type(out["mytuple"]), tuple)

    def test_numeric_and_type_downcast(self):
        class TestInt(int):
            pass
        class TestInt64(np.int64):
            pass
        pytree = {"testint": TestInt(), "testint64": TestInt64()}
        out = self.do_roundtrip(pytree, exact_python_numeric_types=False,
                                downcast_to_base_types=(TestInt, TestInt64))
        self.assertEqual(type(out["testint"]), np.int64)
        self.assertEqual(type(out["testint64"]), np.int64)

    def test_custom_types(self):
        pytree = {
            "return_dict": TestCustomTypeReturnDict(3),
            "return_custom": TestCustomTypeReturnField(TestCustomTypeReturnDict(6)),
        }
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True)

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
            (23, 13): np.arange(34),

            # the reason why this works out of the box
            # is because the return value of jaxon type
            # can never be a simple atom (because it is a container)
            # and always must create a group
            TestCustomTypeReturnField((324, 34)): 24,
            TestCustomDataclass(234, "sdf"): "oasfd",
            TestObjectForDill(): "nksdfnk"
        }
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True, allow_dill=True)

    def test_nested_type_conversion(self):
        pytree = {
            TestCustomTypeReturnField(TestCustomTypeReturnField(TestCustomDataclass(234, "sdf"))):
            TestCustomTypeReturnField(TestCustomTypeReturnField(TestCustomDataclass(34, "sdf43")))
        }
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True)

    def test_single_big_key_value(self):
        pytree = {self.rand_string(42, 1000000), "val"}
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True)

    def test_multi_big_key_value(self):
        pytree = {self.rand_string(i, 100000): i for i in range(10)}
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True)

    def test_custom_dataclass(self):
        pytree = {TestCustomDataclass(213): TestCustomDataclass(TestCustomDataclass(21), "jkk")}
        self.run_roundtrip_test(pytree, exact_python_numeric_types=True)

    def test_by_fuzzing(self):
        random.seed(42)
        for _ in range(100):
            self.run_roundtrip_test(build_fuzz_tree(0, 6), exact_python_numeric_types=True)


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
            class Custom:
                pass
            save(fp, Custom())

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
