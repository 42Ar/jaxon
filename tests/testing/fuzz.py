"""
constants.py

Builds random pytrees for fuzzing tests.

Author
------
Frank Hermann
"""

import jax.numpy as jnp
import numpy as np
from numpy.random import Generator
from jaxon import PyTree
from .classes import CustomTypeReturnDict, ObjectForDill, CustomTypeReturnField, CustomDataclass
from .data import TEST_JAXON_ATOMIC, UNHASHABLE_ATOMIC_TYPES, TEST_NUMPY_ARRAY_VALUES, \
    TEST_JAX_ARRAY_DTYPES, get_jax_array_values, SPECIAL_CHARS, SPECIAL_BYTES


FUZZ_CHARS = (*SPECIAL_CHARS, "A")
FUZZ_BYTES = (*SPECIAL_BYTES, b"A")


def _choice(rng: Generator, seq):
    return seq[rng.integers(len(seq))]


def rand_str(rng: Generator, n: int) -> str:
    return "".join(_choice(rng, FUZZ_CHARS) for _ in range(n))


def rand_bytes(rng: Generator, n: int) -> bytes:
    return b"".join(_choice(rng, FUZZ_BYTES) for _ in range(n))


def build_fuzz_tree(rng: Generator, cur_depth: int, max_depth: int, all_objects: list[PyTree],
        only_hashable_objects: list[PyTree], only_hashable: bool = False) -> PyTree:
    if rng.random() < 0.2:
        # insert reference to existing objects
        if only_hashable:
            if only_hashable_objects:
                return _choice(rng, only_hashable_objects)
        else:
            if all_objects:
                return _choice(rng, all_objects)
    if rng.random() < 0.2 and cur_depth < max_depth:
        subtree = build_fuzz_tree(rng, cur_depth, max_depth, all_objects, only_hashable_objects,
                                  only_hashable=only_hashable)
        custom_class = _choice(rng, [ObjectForDill, CustomTypeReturnDict,
                                     CustomTypeReturnField, CustomDataclass])
        pytree = custom_class(subtree)
    elif rng.random() < 0.5 and cur_depth < max_depth:
        # generate container
        if only_hashable:
            container = _choice(rng, [tuple, frozenset])
            pytree = container(build_fuzz_tree(rng, cur_depth + 1, max_depth, all_objects,
                                               only_hashable_objects, only_hashable)
                               for _ in range(rng.integers(5)))
        else:
            container = _choice(rng, [dict, list, tuple, set, frozenset])
            if container is dict:
                pytree = {build_fuzz_tree(rng, cur_depth + 1, max_depth, all_objects,
                                          only_hashable_objects, only_hashable=True):
                          build_fuzz_tree(rng, cur_depth + 1, max_depth, all_objects,
                                          only_hashable_objects, only_hashable=False)
                          for _ in range(rng.integers(5))}
            else:
                pytree = container(build_fuzz_tree(rng, cur_depth + 1, max_depth, all_objects,
                                                   only_hashable_objects,
                                                   container in (set, frozenset))
                                   for _ in range(rng.integers(5)))
    elif only_hashable or rng.random() < 0.3:
        # put atomic type
        test_atomic = TEST_JAXON_ATOMIC
        if only_hashable:
            test_atomic = {t: v for t, v in test_atomic.items() if t not in UNHASHABLE_ATOMIC_TYPES}
        dtype, examples = _choice(rng, tuple(test_atomic.items()))
        if dtype is str:
            return rand_str(rng, int(rng.integers(5)))
        if dtype is np.str_:
            return np.str_(rand_str(rng, int(rng.integers(5))))
        if dtype is bytes:
            return rand_bytes(rng, int(rng.integers(5)))
        if dtype is bytearray:
            return bytearray(rand_bytes(rng, int(rng.integers(5))))
        if dtype is np.bytes_:
            return np.bytes_(rand_bytes(rng, int(rng.integers(5))))
        if dtype is np.void:
            return np.void(rand_bytes(rng, int(rng.integers(5))))
        pytree = _choice(rng, examples)
    elif rng.random() < 0.5:
        # put numpy array
        dtype, examples = _choice(rng, tuple(TEST_NUMPY_ARRAY_VALUES.items()))
        value = _choice(rng, examples)
        if rng.random() < 0.1:
            pytree = np.array(value, dtype=dtype)
        else:
            pytree = np.array([value], dtype=dtype)
    else:
        # put jax array
        dtype = _choice(rng, TEST_JAX_ARRAY_DTYPES)
        value = get_jax_array_values(dtype)
        if rng.random() < 0.1:
            pytree = jnp.array(value, dtype=dtype)
        else:
            pytree = jnp.array([value], dtype=dtype)
    all_objects.append(pytree)
    if only_hashable:
        only_hashable_objects.append(pytree)
    return pytree


def fuzz_tree_generator(n: int):
    ss = np.random.SeedSequence(0)
    for i, seed in enumerate(ss.spawn(n)):
        print(f"i={i} " + "="*100)
        rng = np.random.default_rng(seed)
        yield build_fuzz_tree(rng, 0, 20, [], [])
