"""
constants.py

Builds random pytrees for fuzzing tests.

Author
------
Frank Hermann
"""

import random
import jax.numpy as jnp
import numpy as np
from jaxon import PyTree
from .classes import CustomDataclass
from .data import TEST_JAXON_ATOMIC, UNHASHABLE_ATOMIC_TYPES, TEST_NUMPY_ARRAY_VALUES, \
    TEST_JAX_ARRAY_DTYPES, get_jax_array_values


def build_fuzz_tree(cur_depth: int,
                    max_depth: int,
                    all_objects: list[PyTree],
                    only_hashable_objects: list[PyTree],
                    only_hashable: bool = False) -> PyTree:
    if random.random() < 0.2:
        # insert reference to existing objects
        if only_hashable:
            if only_hashable_objects:
                return random.choice(only_hashable_objects)
        else:
            if all_objects:
                return random.choice(all_objects)
    if random.random() < 0.2 and cur_depth < max_depth:
        # put TestCustomDataclass with one two random subtree members
        subtree = build_fuzz_tree(cur_depth, max_depth, all_objects, only_hashable_objects,
                                  only_hashable=only_hashable)
        if random.random() < 0.8:
            pytree = CustomDataclass(subtree)
        else:
            subtree2 = build_fuzz_tree(cur_depth, max_depth, all_objects, only_hashable_objects,
                                       only_hashable=only_hashable)
            pytree = CustomDataclass(subtree, subtree2)
    elif random.random() < 0.5 and cur_depth < max_depth:
        # generate container
        if only_hashable:
            container = random.choice([tuple, frozenset])
            pytree = container(build_fuzz_tree(cur_depth + 1, max_depth, all_objects,
                                               only_hashable_objects, only_hashable)
                               for _ in range(random.randint(0, 5)))
        else:
            container = random.choice([dict, set, list, tuple, frozenset])
            if container is dict:
                pytree = {build_fuzz_tree(cur_depth + 1, max_depth, all_objects,
                                          only_hashable_objects, only_hashable=True):
                          build_fuzz_tree(cur_depth + 1, max_depth, all_objects,
                                          only_hashable_objects, only_hashable=False)
                          for _ in range(random.randint(0, 5))}
            else:
                pytree = container(build_fuzz_tree(cur_depth + 1, max_depth, all_objects,
                                                   only_hashable_objects,
                                                   container in (set, frozenset))
                                   for _ in range(random.randint(0, 5)))
    elif only_hashable or random.random() < 0.3:
        # put atomic type
        test_atomic = TEST_JAXON_ATOMIC
        if only_hashable:
            test_atomic = {t: v for t, v in test_atomic.items() if t not in UNHASHABLE_ATOMIC_TYPES}
        dtype, examples = random.choice(tuple(test_atomic.items()))
        if only_hashable and np.issubdtype(dtype, np.inexact):
            # filter to avoid having nan and inf values in sets or dict keys
            examples = tuple(v for v in examples if np.isfinite(v))
        pytree = random.choice(examples)
    elif random.random() < 0.5:
        # put numpy array
        dtype, examples = random.choice(tuple(TEST_NUMPY_ARRAY_VALUES.items()))
        value = random.choice(examples)
        if random.random() < 0.1:
            pytree = np.array(value, dtype=dtype)
        else:
            pytree = np.array([value], dtype=dtype)
    else:
        # put jax array
        dtype = random.choice(TEST_JAX_ARRAY_DTYPES)
        value = get_jax_array_values(dtype)
        if random.random() < 0.1:
            pytree = jnp.array(value, dtype=dtype)
        else:
            pytree = jnp.array([value], dtype=dtype)
    all_objects.append(pytree)
    if only_hashable:
        only_hashable_objects.append(pytree)
    return pytree
