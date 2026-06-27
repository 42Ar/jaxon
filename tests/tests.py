"""
tests.py

Contains tests for the core functionality implemented in jaxon.

Author
------
Frank Hermann
"""

from typing import Any, override
import sys
import tempfile
import random
import string
import pytest
from pathlib import Path
from dataclasses import dataclass, make_dataclass, fields, field
import jax.numpy as jnp
import numpy as np
import h5py
from jaxon import load, save, CircularPyTreeException, JAXON_NP_NUMERIC_TYPES, JAXON_NOT_LOADED, \
    JaxonStorageHints, JAXON_ROOT_GROUP_KEY, PyTree, JaxonFormatWarning, has_common_prefix, \
    JAXON_PY_NUMERIC_TYPES
from .test_util import tree_equal, JaxonPyTreeTestNode


TEST_TYPES = (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
              np.uint64, np.float16, np.float32, np.float64, np.bool_)
TEST_TYPES_FOR_COMPLEX = (np.float32, np.float64)


class _TestObjectForDill(JaxonPyTreeTestNode):
    a = 0.5
    
    def __hash__(self) -> int:
        return hash(self.a)

    @override
    def children(self) -> tuple:
        return (self.a,)


class _TestCustomTypeReturnDict(JaxonPyTreeTestNode):
    def __init__(self, a):
        self.a = a

    def from_jaxon(self, jaxon):
        self.a = jaxon["a"]

    def to_jaxon(self):
        return {"a": self.a}

    @override
    def children(self) -> tuple:
        return (self.a,)


class _TestCustomTypeReturnField(JaxonPyTreeTestNode):
    def __init__(self, obj):
        self.obj = obj

    def from_jaxon(self, jaxon):
        self.obj = jaxon

    def to_jaxon(self):
        return self.obj

    @override
    def children(self) -> tuple:
        return (self.obj,)

    def __hash__(self):
        return hash(self.obj)


@dataclass
class _TestCustomDataclass(JaxonPyTreeTestNode):
    a: Any
    b: Any = 345774

    def __hash__(self):
        return hash((self.a, self.b))

    @override
    def children(self) -> tuple:
        return (self.a, self.b)


def build_fuzz_tree(cur_depth: int, max_depth: int, all_objects: list[PyTree],
        only_hashable_objects: list[PyTree], only_hashable: bool = False) -> PyTree:
    if random.random() < 0.2:
        if only_hashable:
            if only_hashable_objects:
                return random.choice(only_hashable_objects)
        else:
            if all_objects:
                return random.choice(all_objects)
    if random.random() < 0.2:
        subtree = build_fuzz_tree(cur_depth, max_depth, all_objects, only_hashable_objects,
            only_hashable=only_hashable)
        if random.random() < 0.8:
            pytree = _TestCustomDataclass(subtree)
        else:
            subtree2 = build_fuzz_tree(cur_depth, max_depth, all_objects, only_hashable_objects,
                only_hashable=only_hashable)
            pytree = _TestCustomDataclass(subtree, subtree2)
    elif random.random() < 0.5 and cur_depth < max_depth:
        if not only_hashable and random.random() < 0.1:
            container = random.choice((list,))
            pytree = container([build_fuzz_tree(cur_depth + 1, max_depth, all_objects,
                                only_hashable_objects, only_hashable=True)
                               for _ in range(random.randint(0, 5))])
        elif not only_hashable and random.random() < 0.4:
            pytree =  {build_fuzz_tree(cur_depth + 1, max_depth, all_objects,
                                       only_hashable_objects, only_hashable=True):
                       build_fuzz_tree(cur_depth + 1, max_depth, all_objects,
                                       only_hashable_objects, only_hashable=False)
                       for _ in range(random.randint(0, 5))}
        else:
            pytree = tuple(build_fuzz_tree(cur_depth + 1, max_depth, all_objects,
                                           only_hashable_objects, only_hashable)
                          for _ in range(random.randint(0, 5)))
    else:
        if random.random() < 0.3:
            if only_hashable:
                pytree = 3984789438723
            else:
                pytree =  np.arange(3)
        else:
            pytree = "".join(random.choice("\\/'\0ä") for _ in range(random.randint(0, 30)))
    all_objects.append(pytree)
    if only_hashable:
        only_hashable_objects.append(pytree)
    return pytree


def do_roundtrip(pytree, exact_python_numeric_types, allow_dill=False,
                 downcast_to_base_types=None):
    with tempfile.TemporaryFile() as fp:
        save(fp, pytree, exact_python_numeric_types=exact_python_numeric_types,
             downcast_to_base_types=downcast_to_base_types, allow_dill=allow_dill)
        return load(fp, allow_dill=allow_dill)


def run_roundtrip_test(pytree, exact_python_numeric_types, allow_dill=False):
    loaded = do_roundtrip(pytree, exact_python_numeric_types, allow_dill)
    py_to_np_types = tuple()
    if not exact_python_numeric_types:
        py_to_np_types = JAXON_PY_NUMERIC_TYPES
    tree_equal(loaded, pytree, tuple(), py_to_np_types)
    return loaded


def rand_string(seed, n):
    random.seed(seed)
    special = ["'", '"', "\0", "\r", "\n", "ä", "ö", "ü", "ß", ":", "\\"]
    return "".join(random.choices(list(string.ascii_uppercase) + special, k=n))


def test_roundtrip_simple_types():
    pytree = {
        "complex": 1j + 5,
        "complex_pure_imaginary": 1j,
        "bool": True,
        "bool2": False,
        "None": None,
        "string": "string",
        "string_with_quotes1": "'",
        "string_with_quotes2": '"',
        "string_with_quotes3": '"\'',
        "string_with_quotes_and_slashes": '/',
        "string_with_quotes_and_backslashes": '/bk/asd\\/sd\\\\//\\/',
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
        "npbool": np.bool_(3465.34),
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
        run_roundtrip_test(pytree, exact_python_numeric_types)


def test_roundtrip_platform_specific_scalar_types():
    """np.longdouble and np.clongdouble scalars must round-trip on every platform."""
    pytree = {
        "longdouble": np.longdouble(1.5),
        "clongdouble": np.clongdouble(1.5 + 2.5j),
        "longdouble_array": np.array([1.0, 2.0], dtype=np.longdouble),
        "clongdouble_array": np.array([1.0+2j, 3.0+4j], dtype=np.clongdouble),
    }
    run_roundtrip_test(pytree, exact_python_numeric_types=True)


def test_roundtrip_arrays():
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
        "complex": [random_complex(scalar_type) for scalar_type in TEST_TYPES_FOR_COMPLEX],
        "complex_jax": [jnp.array(random_complex(scalar_type)) for scalar_type in TEST_TYPES_FOR_COMPLEX]
    }
    for exact_python_numeric_types in (False, True):
        run_roundtrip_test(pytree, exact_python_numeric_types)


def test_roundtrip_trivial_roots():
    for exact_python_numeric_types in (False, True):
        run_roundtrip_test(1, exact_python_numeric_types)
        run_roundtrip_test(None, exact_python_numeric_types)
        run_roundtrip_test({}, exact_python_numeric_types)
        run_roundtrip_test({"a": 345}, exact_python_numeric_types)
        run_roundtrip_test([], exact_python_numeric_types)
        run_roundtrip_test([3], exact_python_numeric_types)
        run_roundtrip_test(b"dfuikfhk\0\0ufs", exact_python_numeric_types)
        run_roundtrip_test(np.arange(2), exact_python_numeric_types)
        run_roundtrip_test(jnp.arange(2), exact_python_numeric_types)


def test_roundtrip_dill_object_at_root():
    run_roundtrip_test(_TestObjectForDill(), False, allow_dill=True)


def test_roundtrip_dill_objects_in_container():
    pytree = [{"adssd": _TestObjectForDill()}, _TestObjectForDill()]
    for exact_python_numeric_types in (False, True):
        run_roundtrip_test(pytree, exact_python_numeric_types, allow_dill=True)


def test_roundtrip_numeric_type_conversion():
    pytree = {"int": 3, "float": 45.4, "complex": 4j + 4, "bool": True}
    out = run_roundtrip_test(pytree, exact_python_numeric_types=False)
    assert type(out["int"]) == np.int64
    assert type(out["float"]) == np.float64
    assert type(out["complex"]) == np.complex128
    assert type(out["bool"]) == np.bool_


def test_roundtrip_type_downcast():
    class TestInt(int):
        pass
    class TestInt64(np.int64):
        pass
    pytree = {"testint": TestInt(), "testint64": TestInt64()}
    out = do_roundtrip(pytree, exact_python_numeric_types=True,
                    downcast_to_base_types=(TestInt, TestInt64))
    assert type(out["testint"]) == int
    assert type(out["testint64"]) == np.int64


def test_roundtrip_container_type_downcast():
    class TestDict(dict):
        pass
    class TestList(list):
        pass
    class TestTuple(tuple):
        pass
    pytree = TestDict({"mylist": TestList([12, 231, TestList(["ads"])]),
                     "mytuple": TestTuple((324, 234, "df"))})
    out = do_roundtrip(pytree, exact_python_numeric_types=True,
                    downcast_to_base_types=[TestDict, TestList, TestTuple])
    assert type(out) == dict
    assert type(out["mylist"]) == list
    assert type(out["mytuple"]) == tuple


def test_roundtrip_numeric_and_type_downcast():
    class TestInt(int):
        pass
    class TestInt64(np.int64):
        pass
    pytree = {"testint": TestInt(), "testint64": TestInt64()}
    out = do_roundtrip(pytree, exact_python_numeric_types=False,
                    downcast_to_base_types=(TestInt, TestInt64))
    assert type(out["testint"]) == np.int64
    assert type(out["testint64"]) == np.int64


def test_roundtrip_custom_types():
    pytree = {
        "return_dict": _TestCustomTypeReturnDict(3),
        "return_custom": _TestCustomTypeReturnField(_TestCustomTypeReturnDict(6)),
    }
    run_roundtrip_test(pytree, exact_python_numeric_types=True)


def test_roundtrip_single_big_attr_value():
    pytree = rand_string(42, 1000000)
    run_roundtrip_test(pytree, exact_python_numeric_types=True)


def test_roundtrip_multi_big_attr_value():
    pytree = [rand_string(i, 100000) for i in range(10)]
    run_roundtrip_test(pytree, exact_python_numeric_types=True)


def test_roundtrip_nonstring_dict_keys():
    pytree = {
        0: "ksdnkf",
        1: "asd",
        234: 5,
        (34, 234): 8,
        "sfddf": "dfs",
        (23, 13): np.arange(34),
        _TestCustomTypeReturnField((324, 34)): 24,
        _TestCustomDataclass(234, "sdf"): "oasfd",
        _TestObjectForDill(): "nksdfnk"
    }
    run_roundtrip_test(pytree, exact_python_numeric_types=True, allow_dill=True)


def test_roundtrip_nested_type_conversion():
    pytree = {
        _TestCustomTypeReturnField(_TestCustomTypeReturnField(_TestCustomDataclass(234, "sdf"))):
        _TestCustomTypeReturnField(_TestCustomTypeReturnField(_TestCustomDataclass(34, "sdf43")))
    }
    run_roundtrip_test(pytree, exact_python_numeric_types=True)


def test_roundtrip_single_big_key_value():
    pytree = {rand_string(42, 1000000): 42}
    run_roundtrip_test(pytree, exact_python_numeric_types=True)


def test_roundtrip_multi_big_key_value():
    pytree = {rand_string(i, 100000): i for i in range(10)}
    run_roundtrip_test(pytree, exact_python_numeric_types=True)


def test_roundtrip_custom_dataclass():
    pytree = {_TestCustomDataclass(213): _TestCustomDataclass(_TestCustomDataclass(21), "jkk")}
    run_roundtrip_test(pytree, exact_python_numeric_types=True)


def test_roundtrip_by_fuzzing():
    random.seed(42)
    for _ in range(200):
        pytree = build_fuzz_tree(0, 20, [], [])
        run_roundtrip_test(pytree, exact_python_numeric_types=True)


def test_error_circular_reference_detection():
    def trigger_circular_reference_exception():
        pytree = {}
        pytree["a"] = pytree
        with tempfile.TemporaryFile() as fp:
            save(fp, pytree)

    with pytest.raises(CircularPyTreeException):
        trigger_circular_reference_exception()


def test_error_unsupported_object():
    def trigger_unsupported_type_exception():
        with tempfile.TemporaryFile() as fp:
            class Custom:
                pass
            save(fp, Custom())

    with pytest.raises(TypeError):
        trigger_unsupported_type_exception()


def test_intro_store_in_dataclass():
    pytree = {"attribute": np.zeros(10), "dataset": np.zeros(10)}
    with tempfile.TemporaryFile() as fp:
        save(fp, pytree, storage_hints=[(pytree["dataset"], JaxonStorageHints(True))])
        with h5py.File(fp, 'r') as file:
            assert "'dataset'" in list(file[JAXON_ROOT_GROUP_KEY])
            assert len(list(file[JAXON_ROOT_GROUP_KEY])) == 1
            assert "'attribute'" not in list(file[JAXON_ROOT_GROUP_KEY])


def test_truncate_fp():
    def do_test_truncate(path_or_fp):
        save(path_or_fp, {"a": 3, "b": 2})
        save(path_or_fp, {"a": 3})
        assert load(path_or_fp) == {"a": 3}

    with tempfile.TemporaryFile() as fp:
        do_test_truncate(fp)


def test_truncate_real_file():
    def do_test_truncate(path_or_fp):
        save(path_or_fp, {"a": 3, "b": 2})
        save(path_or_fp, {"a": 3})
        assert load(path_or_fp) == {"a": 3}

    with tempfile.TemporaryDirectory() as tmpdirname:
        do_test_truncate(Path(tmpdirname) / "t.hdf5")


def test_custom_marshaler():
    with tempfile.TemporaryFile() as fp:
        class MyCustomClass:
            def __init__(self, a, b):
                self.a = a
                self.b = b
            
            def __eq__(self, other):
                return self.a == other.a and self.b == other.b

        def my_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
            if isinstance(pytree, MyCustomClass):
                return "mycustomtypeid", {"a": pytree.a, "b": pytree.b}
            return None

        pytree = {"g": MyCustomClass(MyCustomClass(None, 3), 0), "f": 123}
        save(fp, pytree, custom_marshalers=(my_marshaler,))

        def my_unmarshaler(qualname: str, pytree: PyTree) -> PyTree | None:
            if qualname == "mycustomtypeid":
                return MyCustomClass(pytree["a"], pytree["b"])
            return None

        loaded_pytree = load(fp, custom_unmarshalers=(my_unmarshaler,))
        tree_equal(loaded_pytree, pytree)


def test_relaxed_raise_if_missing_fields_are_not_allowed():
    def run_test_allow_missing_fields(**kwargs):
        with tempfile.TemporaryFile() as fp:
            Dynamic = make_dataclass(
                "Dynamic",
                [("a", int), ("b", float)],
            )
            module = sys.modules[__name__]
            setattr(module, "Dynamic", Dynamic)
            pytree = Dynamic(a=123, b=2.0)
            assert len(fields(pytree)) == 2
            save(fp, pytree)
            Dynamic = make_dataclass(
                "Dynamic",
                [("a", int)],
            )
            setattr(module, "Dynamic", Dynamic)
            loaded_pytree = load(fp, **kwargs)
            assert loaded_pytree.a == pytree.a
            assert len(fields(loaded_pytree)) == 1

    with pytest.raises(ValueError):
        run_test_allow_missing_fields(allow_missing_fields=False)


def test_relaxed_allow_missing_fields():
    def run_test_allow_missing_fields(**kwargs):
        with tempfile.TemporaryFile() as fp:
            Dynamic = make_dataclass(
                "Dynamic",
                [("a", int), ("b", float)],
            )
            module = sys.modules[__name__]
            setattr(module, "Dynamic", Dynamic)
            pytree = Dynamic(a=123, b=2.0)
            assert len(fields(pytree)) == 2
            save(fp, pytree)
            Dynamic = make_dataclass(
                "Dynamic",
                [("a", int)],
            )
            setattr(module, "Dynamic", Dynamic)
            loaded_pytree = load(fp, **kwargs)
            assert loaded_pytree.a == pytree.a
            assert len(fields(loaded_pytree)) == 1

    with pytest.warns(JaxonFormatWarning):
        run_test_allow_missing_fields(allow_missing_fields=True)


def test_relaxed_raise_if_unknown_fields_are_not_allowed():
    def run_test_allow_unknown_fields(**kwargs):
        with tempfile.TemporaryFile() as fp:
            Dynamic = make_dataclass(
                "Dynamic",
                [("existing", int)],
            )
            module = sys.modules[__name__]
            setattr(module, "Dynamic", Dynamic)
            pytree = Dynamic(existing=123)
            assert len(fields(pytree)) == 1
            save(fp, pytree)
            Dynamic = make_dataclass(
                "Dynamic",
                [("existing", int),
                 ("missing_mandatory", float),
                 ("missing_default", float, field(default=2)),
                 ("missing_default_factory", float, field(default_factory=lambda: 3)),
                 ("missing_default_factory_no_init", float, field(default_factory=lambda: 5, init=False))],
            )
            setattr(module, "Dynamic", Dynamic)
            loaded_pytree = load(fp, **kwargs)
            assert loaded_pytree.existing == 123
            assert loaded_pytree.missing_mandatory is JAXON_NOT_LOADED
            assert loaded_pytree.missing_default == 2
            assert loaded_pytree.missing_default_factory == 3
            assert loaded_pytree.missing_default_factory_no_init == 5

    with pytest.raises(ValueError):
        run_test_allow_unknown_fields(allow_unknown_fields=False)


def test_relaxed_allow_unknown_fields():
    def run_test_allow_unknown_fields(**kwargs):
        with tempfile.TemporaryFile() as fp:
            Dynamic = make_dataclass(
                "Dynamic",
                [("existing", int)],
            )
            module = sys.modules[__name__]
            setattr(module, "Dynamic", Dynamic)
            pytree = Dynamic(existing=123)
            assert len(fields(pytree)) == 1
            save(fp, pytree)
            Dynamic = make_dataclass(
                "Dynamic",
                [("existing", int),
                 ("missing_mandatory", float),
                 ("missing_default", float, field(default=2)),
                 ("missing_default_factory", float, field(default_factory=lambda: 3)),
                 ("missing_default_factory_no_init", float, field(default_factory=lambda: 5, init=False))],
            )
            setattr(module, "Dynamic", Dynamic)
            loaded_pytree = load(fp, **kwargs)
            assert loaded_pytree.existing == 123
            assert loaded_pytree.missing_mandatory is JAXON_NOT_LOADED
            assert loaded_pytree.missing_default == 2
            assert loaded_pytree.missing_default_factory == 3
            assert loaded_pytree.missing_default_factory_no_init == 5

    with pytest.warns(JaxonFormatWarning):
        run_test_allow_unknown_fields(allow_unknown_fields=True)


def test_filter_filtering():
    with tempfile.TemporaryFile() as fp:
        save(fp, {"a": {"a": 2}, "b": [_TestCustomDataclass({"a": 5}, 3), "c"]})
        loaded = load(fp, load_filter=lambda path: has_common_prefix(path, ("a",)))
        tree_equal(loaded, {"a": {"a": 2}, "b": JAXON_NOT_LOADED})
        loaded = load(fp, load_filter=lambda path: has_common_prefix(path, ("b", 1)))
        tree_equal(loaded, {'a': JAXON_NOT_LOADED, 'b': [JAXON_NOT_LOADED, 'c']})
        loaded = load(fp, load_filter=lambda path: has_common_prefix(path, ("b", 0, "b")))
        tree_equal(loaded, {'a': JAXON_NOT_LOADED, 'b': [_TestCustomDataclass(a=JAXON_NOT_LOADED, b=3), JAXON_NOT_LOADED]})
        loaded = load(fp, load_filter=lambda path: False)
        tree_equal(loaded, JAXON_NOT_LOADED)
        loaded = load(fp, load_filter=lambda path: has_common_prefix(path, ("b", 0)))
        tree_equal(loaded, {'a': JAXON_NOT_LOADED, 'b': [_TestCustomDataclass(a={'a': 5}, b=3), JAXON_NOT_LOADED]})


def test_reference_correct_ref_recovery():
    a = np.array([3, 4])
    b = np.array([8, 4])
    g = (a, a, b)
    pytree = {"a": a, "b": g, "c": b, "d": (g,)}
    with tempfile.TemporaryFile() as fp:
        save(fp, pytree)
        loaded = load(fp)
        tree_equal(pytree, loaded)
        assert loaded["a"] is loaded["b"][0]
        assert loaded["a"] is loaded["b"][1]
        assert loaded["c"] is loaded["b"][2]
        assert loaded["b"] is loaded["d"][0]


def test_reference_correct_ref_recovery_partially_loaded():
    a = np.array([3, 4])
    b = np.array([8, 4])
    g = (a, a, b)
    pytree = {"a": a, "b": g, "c": b, "d": (g,)}
    with tempfile.TemporaryFile() as fp:
        save(fp, pytree)
        loaded = load(fp, load_filter=lambda path: has_common_prefix(path, ("a",)) or has_common_prefix(path, ("b", 0)))
        tree_equal(loaded, {'a': a, 'b': (a, JAXON_NOT_LOADED, JAXON_NOT_LOADED), 'c': JAXON_NOT_LOADED, 'd': JAXON_NOT_LOADED})