"""
tests.py

Contains tests of the core functionality (save/load api).

Author
------
Frank Hermann
"""

from typing import override
from dataclasses import make_dataclass, fields, field
from pathlib import Path
import sys
import tempfile
import random
import pytest
import jax.numpy as jnp
import numpy as np
from numpy.random import default_rng, SeedSequence
from jaxon import load, save, CircularPyTreeError, JaxonNotLoaded, PyTree, \
    JaxonFormatWarning, has_common_prefix, JAXON_PY_NUMERIC_TYPES, JAXON_PY_CONTAINER_TYPES, \
    JaxonTypeWarning, JaxonTypeError
from jaxon._common import JaxonFormatError
from .testing.classes import CustomDataclass, ObjectForDill, CustomTypeReturnDict, \
    CustomTypeReturnTuple
from .testing.data import TEST_JAXON_ATOMIC, TEST_NUMPY_ARRAY_VALUES, TEST_JAX_ARRAY_DTYPES, \
    get_jax_array_values
from .testing.tree_equal import assert_tree_equal, PyTreeTestNode
from .testing.fuzz import fuzz_tree_generator, rand_str


def do_roundtrip(pytree, exact_python_numeric_types=True, allow_dill=False,
                 downcast_to_base_types=None):
    with tempfile.TemporaryFile() as fp:
        save(fp, pytree, exact_python_numeric_types=exact_python_numeric_types,
             downcast_to_base_types=downcast_to_base_types, allow_dill=allow_dill)
        return load(fp, allow_dill=allow_dill)


def run_roundtrip_test(pytree, exact_python_numeric_types=True, allow_dill=False,
                       downcast_to_base_types=tuple()):
    loaded = do_roundtrip(pytree, exact_python_numeric_types, allow_dill, downcast_to_base_types)
    py_to_np_types = tuple()
    if not exact_python_numeric_types:
        py_to_np_types = JAXON_PY_NUMERIC_TYPES
    assert_tree_equal(pytree, loaded, downcast_to_base_types, py_to_np_types)
    return loaded


def test_atomic_types():
    pytree = {tp.__name__: v for tp, v in TEST_JAXON_ATOMIC.items()}
    for exact_python_numeric_types in (True, False):
        run_roundtrip_test(pytree, exact_python_numeric_types)


def test_nd_numpy_arrays():
    for shape in (tuple(), (0,), (1,), (3,), (2, 3)):
        pytree = {tp.__name__: tuple(np.full(shape, t) for t in v)
                               for tp, v in TEST_NUMPY_ARRAY_VALUES.items()}
        for exact_python_numeric_types in (True, False):
            run_roundtrip_test(pytree, exact_python_numeric_types)


def test_nd_jax_arrays():
    random.seed(0)
    for shape in (tuple(), (0,), (1,), (3,), (2, 3)):
        for dtype in TEST_JAX_ARRAY_DTYPES:
            value = random.choice(get_jax_array_values(dtype))
            run_roundtrip_test(jnp.full(shape, value))


def test_jax_scalars():
    for dtype in TEST_JAX_ARRAY_DTYPES:
        for value in get_jax_array_values(dtype):
            print(value, dtype)
            run_roundtrip_test(jnp.array(value, dtype=dtype))


def test_trivial_roots():
    for container_type in JAXON_PY_CONTAINER_TYPES:
        run_roundtrip_test(container_type())
    for exact_python_numeric_types in (False, True):
        for examples in TEST_JAXON_ATOMIC.values():
            for e in examples:
                run_roundtrip_test(e, exact_python_numeric_types)
    run_roundtrip_test(ObjectForDill(3), allow_dill=True)
    # arrays have already been tested as root nodes


def test_dill_objects_in_container():
    pytree = [{"0": ObjectForDill(False)}, ObjectForDill({"a": 3})]
    for exact_python_numeric_types in (False, True):
        run_roundtrip_test(pytree, exact_python_numeric_types, allow_dill=True)


def test_roundtrip_numeric_type_conversion_explicit():
    """already covered by many tests using TEST_JAXON_ATOMIC, but here more explicitly"""
    pytree = {"int": 3, "float": 45.4, "complex": 4j + 4, "bool": True}
    loaded = run_roundtrip_test(pytree, exact_python_numeric_types=False)
    assert type(loaded["int"]) is np.int64
    assert type(loaded["float"]) is np.float64
    assert type(loaded["complex"]) is np.complex128
    assert type(loaded["bool"]) is np.bool_


def test_type_downcast():
    class TestInt(int):
        pass
    class TestInt64(np.int64):
        pass
    pytree = {"testint": TestInt(), "testint64": TestInt64()}
    loaded = run_roundtrip_test(pytree, downcast_to_base_types=(TestInt, TestInt64))
    assert type(loaded["testint"]) is int
    assert type(loaded["testint64"]) is np.int64


def test_container_type_downcast():
    class TestDict(dict):
        pass
    class TestList(list):
        pass
    class TestTuple(tuple):
        pass
    pytree = TestDict({"mylist": TestList([1, 2, TestList(["1"])]),
                       "mytuple": TestTuple((3, 4, "1"))})
    out = run_roundtrip_test(pytree, downcast_to_base_types=[TestDict, TestList, TestTuple])
    assert type(out) == dict
    assert type(out["mylist"]) == list
    assert type(out["mytuple"]) == tuple


def test_type_downcast_and_int_conversion():
    class TestInt64(int):
        pass
    pytree = {"testint": TestInt64()}
    loaded = run_roundtrip_test(pytree, exact_python_numeric_types=False,
                                downcast_to_base_types=(TestInt64,))
    assert type(loaded["testint"]) is np.int64  # explicit check


def test_custom_types():
    pytree = {
        "return_dict": CustomTypeReturnDict(3),
        "return_custom": CustomTypeReturnTuple(CustomTypeReturnDict(6)),
    }
    run_roundtrip_test(pytree)


def test_single_big_attr_value():
    run_roundtrip_test(rand_str(default_rng(), 1000000))


def test_multi_big_attr_value():
    ss = SeedSequence(0)
    pytree = [rand_str(default_rng(seed), 100000) for seed in ss.spawn(10)]
    run_roundtrip_test(pytree)


def test_nonstring_dict_keys():
    pytree = {
        0: "0",
        1: "1",
        np.nan: 5,
        (1, 2): 8,
        "1": "2",
        (3, 4): np.arange(42),
        CustomTypeReturnTuple((324, 34)): 24,
        CustomDataclass(234, "a"): "v",
        ObjectForDill(1): "x"
    }
    run_roundtrip_test(pytree, allow_dill=True)


def test_nested_type_conversion():
    pytree = {
        CustomTypeReturnTuple(CustomTypeReturnTuple(CustomDataclass(1, "4"))):
        CustomDataclass(CustomTypeReturnTuple(CustomTypeReturnTuple(1)))
    }
    run_roundtrip_test(pytree)


def test_single_big_key_value():
    pytree = {rand_str(default_rng(0), 1000000): 42}
    run_roundtrip_test(pytree)


def test_multi_big_key_value():
    pytree = {rand_str(default_rng(0), 100000): i for i in range(10)}
    run_roundtrip_test(pytree)


def test_custom_dataclass():
    pytree = {CustomDataclass(1): CustomDataclass(CustomDataclass(2), "a")}
    run_roundtrip_test(pytree)


def test_reference_with_escape_symbols():
    a = []
    pytree = {"\\": a, "d": a}
    run_roundtrip_test(pytree)


def test_fuzzing():
    for pytree in fuzz_tree_generator(1000):
        run_roundtrip_test(pytree, allow_dill=True)


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
        class MyCustomClass(PyTreeTestNode):
            def __init__(self, a, b):
                self.a = a
                self.b = b

            @override
            def children(self) -> tuple:
                return (self.a, self.b)

        def my_marshaler(pytree: PyTree) -> tuple[str, PyTree] | None:
            if isinstance(pytree, MyCustomClass):
                return "mycustomtypeid", {"a": pytree.a, "b": pytree.b}
            return None

        pytree = {
            "MyCustomClass": MyCustomClass(MyCustomClass(None, 3), 0),
            "OtherCustomClass": CustomTypeReturnDict(1)
        }
        save(fp, pytree, custom_marshalers=(my_marshaler,))

        def my_unmarshaler(qualname: str, pytree: PyTree) -> PyTree | None:
            if qualname == "mycustomtypeid":
                return MyCustomClass(pytree["a"], pytree["b"])
            return None

        loaded = load(fp, custom_unmarshalers=(my_unmarshaler,))
        assert_tree_equal(pytree, loaded)


def test_load_filter():
    with tempfile.TemporaryFile() as fp:
        save(fp, {"a": {"a": 2}, "b": [CustomDataclass({"a": 5}, 3), "c"]})
        loaded = load(fp, load_filter=lambda path: has_common_prefix(path, ("a",)))
        assert_tree_equal({"a": {"a": 2}, "b": JaxonNotLoaded()}, loaded)
        loaded = load(fp, load_filter=lambda path: has_common_prefix(path, ("b", 1)))
        assert_tree_equal({'a': JaxonNotLoaded(), 'b': [JaxonNotLoaded(), 'c']}, loaded)
        loaded = load(fp, load_filter=lambda path: has_common_prefix(path, ("b", 0, "b")))
        assert_tree_equal({'a': JaxonNotLoaded(), 'b': [CustomDataclass(a=JaxonNotLoaded(), b=3), JaxonNotLoaded()]}, loaded)
        loaded = load(fp, load_filter=lambda path: False)
        assert_tree_equal(JaxonNotLoaded(), loaded)
        loaded = load(fp, load_filter=lambda path: has_common_prefix(path, ("b", 0)))
        assert_tree_equal({'a': JaxonNotLoaded(), 'b': [CustomDataclass(a={'a': 5}, b=3), JaxonNotLoaded()]}, loaded)


def test_reference_recovery_explicitly():
    """references are checked by assert_tree_equal; this is an explicit test"""
    a = np.array([3, 4])
    b = np.array([8, 4])
    g = (a, a, b)
    pytree = {"a": a, "b": g, "c": b, "d": (g,)}
    loaded = run_roundtrip_test(pytree)
    assert loaded["a"] is loaded["b"][0]
    assert loaded["a"] is loaded["b"][1]
    assert loaded["c"] is loaded["b"][2]
    assert loaded["b"] is loaded["d"][0]


def test_reference_recovery_if_partially_loaded():
    a = np.array([3, 4])
    b = np.array([8, 5])
    g = (a, a, b)
    pytree = {"a": a, "b": g, "c": b, "d": (g,)}
    expected_pytree = {
        'a': a,
        'b': (a, JaxonNotLoaded(), JaxonNotLoaded()),
        'c': JaxonNotLoaded(),
        'd': JaxonNotLoaded()
    }
    with tempfile.TemporaryFile() as fp:
        save(fp, pytree)
        loaded = load(fp, load_filter=lambda path: (has_common_prefix(path, ("a",))
                                                    or has_common_prefix(path, ("b", 0))))
        assert_tree_equal(expected_pytree, loaded)


def test_structured_numpy_array():
    x = np.array([((b'A', b'B'), 9, 1.1), ((b'C', b'D'), 3, 7.2)],
        dtype=[('d', [('h', 'V10'), ('i', 'V10')]), ('b', 'i4'), ('g', 'f4')])
    run_roundtrip_test(x)


def test_warn_numpy_array_with_title():
    with pytest.warns(JaxonTypeWarning):
        x = np.array([(b'A', 9, 81.0), (b'B', 3, 27.0)],
            dtype=[(('sd', 'd'), 'V10'), ('b', 'i4'), ('g', 'f4')])
        with tempfile.TemporaryFile() as fp:
            save(fp, x)


def test_raise_numpy_array_with_unsupported_dtype():
    with pytest.raises(JaxonTypeError):
        with tempfile.TemporaryFile() as fp:
            save(fp, np.array([], dtype=np.object_))
    with pytest.raises(JaxonTypeError):
        with tempfile.TemporaryFile() as fp:
            save(fp, np.array(["A"], dtype='U10'))
    with pytest.raises(JaxonTypeError):
        with tempfile.TemporaryFile() as fp:
            save(fp, np.array(["A"], dtype=[('a', [('b', 'U10')])]))


def test_raise_circular_reference():
    def trigger_circular_reference_exception():
        pytree = {}
        pytree["a"] = pytree
        with tempfile.TemporaryFile() as fp:
            save(fp, pytree)
    with pytest.raises(CircularPyTreeError):
        trigger_circular_reference_exception()


def test_raise_unsupported_type():
    with pytest.raises(JaxonTypeError):
        class Custom:
            pass
        with tempfile.TemporaryFile() as fp:
            save(fp, Custom())


def test_raise_and_warn_missing_fields():
    def run_test_missing_fields(**kwargs):
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
            assert type(loaded_pytree.a) is type(pytree.a)
            assert len(fields(loaded_pytree)) == 1
    with pytest.raises(JaxonFormatError):
        run_test_missing_fields(allow_missing_fields=False)
    with pytest.warns(JaxonFormatWarning):
        run_test_missing_fields(allow_missing_fields=True)


def test_raise_and_warn_unknown_fields():
    def run_test_unknown_fields(**kwargs):
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
                 ("missing_default_factory_no_init",
                  float, field(default_factory=lambda: 5, init=False))],
            )
            setattr(module, "Dynamic", Dynamic)
            loaded_pytree = load(fp, **kwargs)
            assert loaded_pytree.existing == 123
            assert type(loaded_pytree.missing_mandatory) is JaxonNotLoaded
            assert loaded_pytree.missing_default == 2
            assert loaded_pytree.missing_default_factory == 3
            assert loaded_pytree.missing_default_factory_no_init == 5
    with pytest.raises(JaxonFormatError):
        run_test_unknown_fields(allow_unknown_fields=False)
    with pytest.warns(JaxonFormatWarning):
        run_test_unknown_fields(allow_unknown_fields=True)
