"""
constants.py

Defines constants and functions to produce test data.

Author
------
Frank Hermann
"""

import random
import string
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp
from jax.extend.core import array_types as jax_array_types


jax.config.update("jax_enable_x64", True)


def is_supported_jax_dtype(dtype):
    try:
        return jnp.empty((), dtype=dtype).dtype == jnp.dtype(dtype)
    except Exception:
        return False


TEST_FLOATS = (float(0), float(1), 1.1, -1.1, float(np.inf), float(-np.inf), float(np.nan))
TEST_COMPLEX = tuple(r + 1j*i for i, r in product(TEST_FLOATS, TEST_FLOATS))
TEST_STRINGS = ("", "'", '"', "''", '""', ":", "\\", "\\\\", "\0", "/", "/A/A/", "\0A\0", "ö")
TEST_BYTES = (b"", b"'", b'"', b"''", b'""', b":", b"\\", b"\\\\", b"\0", b"/", b"/A/A/", b"\0A\0")
TEST_NUMPY_NUMERIC = {
    np.bool_: (np.bool_(True), np.bool_(False)),
    np.int8: (np.int8(0), np.int8(1)),
    np.uint8: (np.uint8(0), np.uint8(1)),
    np.int16: (np.int16(0), np.int16(1)),
    np.uint16: (np.uint16(0), np.uint16(1)),
    np.int32: (np.int32(0), np.int32(1)),
    np.uint32: (np.uint32(0), np.uint32(1)),
    np.int64: (np.int64(0), np.int64(1)),
    np.uint64: (np.uint64(0), np.uint64(1)),

    np.float16: tuple(map(np.float16, TEST_FLOATS)),
    np.float32: tuple(map(np.float32, TEST_FLOATS)),
    np.float64: tuple(map(np.float64, TEST_FLOATS)),

    np.complex64: tuple(map(np.complex64, TEST_COMPLEX)),
    np.complex128: tuple(map(np.complex128, TEST_COMPLEX)),
}
if hasattr(np, "float128"):
    TEST_NUMPY_NUMERIC |= {np.float128: tuple(map(np.float128, TEST_FLOATS))}
if hasattr(np, "complex256"):
    TEST_NUMPY_NUMERIC |= {np.complex256: tuple(map(np.complex256, TEST_COMPLEX))}
TEST_JAXON_ATOMIC = TEST_NUMPY_NUMERIC | {
    np.str_: tuple(map(np.str_, TEST_STRINGS)),
    np.bytes_: tuple(map(np.bytes_, TEST_BYTES)),  # note that this strips trailing zero characters
    np.void: tuple(map(np.void, TEST_BYTES)),

    str: TEST_STRINGS,
    bytes: TEST_BYTES,
    bytearray: tuple(map(bytearray, TEST_BYTES)),

    int: (0, 1, -1, 8734287324643882734672323),
    float: TEST_FLOATS,
    bool: (True, False),
    complex: TEST_COMPLEX,

    type(None): (None,),
    type(Ellipsis): (Ellipsis,),

    range: (range(0), range(10), range(1, 10), range(1, 10, 2))
}
TEST_NUMPY_ARRAY_VALUES = TEST_NUMPY_NUMERIC | {
    # np.str_ is not supported in arrays
    # null bytes in np.bytes_ are not supported in arrays
    # zero length strings are not supported
    np.bytes_: tuple(np.bytes_(x) for x in TEST_BYTES if len(x) > 0 and b"\0" not in x),
    np.void: tuple(np.void(x) for x in TEST_BYTES if len(x) > 0),
}
TEST_JAX_ARRAY_DTYPES = tuple(dtype for dtype in jax_array_types
                              if (dtype.__name__ not in ("ndarray", "TypedNdArray")
                                  and is_supported_jax_dtype(dtype)))
UNHASHABLE_ATOMIC_TYPES = (bytearray, np.void)


def get_jax_array_values(dtype):
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return TEST_COMPLEX
    if jnp.issubdtype(dtype, jnp.inexact):
        return TEST_FLOATS
    return (0, 1)


def rand_string(seed, n):
    random.seed(seed)
    special = ["'", '"', "\0", "\r", "\n", "ä", "ö", "ü", "ß", ":", "\\"]
    return "".join(random.choices(list(string.ascii_uppercase) + special, k=n))
