# AGENTS.md — Jaxon

## Structure

- **Four-module package** under `src/jaxon/`:
  - `_common.py` — constants, type aliases, exceptions, internal data classes, shared utilities
  - `_save.py` — pytree→atom conversion and HDF5 writing; contains `save()`
  - `_load.py` — HDF5 reading and atom→pytree reconstruction; contains `load()`
  - `__init__.py` — public re-exports and `__all__`; the public API is `save()` and `load()`
- **Tests** in `tests/tests.py` (unittest, no pytest), with `tests/test_util.py` providing `tree_equal()` and `JaxonPyTreeTestNode`.

## Commands

```bash
# Run all tests (no installation required)
PYTHONPATH=src python -m unittest tests.tests

# Run a single test class
PYTHONPATH=src python -m unittest tests.tests.RoundtripTests

# Run a single test method
PYTHONPATH=src python -m unittest tests.tests.RoundtripTests.test_simple_types

# Build
python -m build
```

## Key conventions

- **No CI/CD, no linter, no formatter, no type checker** — none are configured.
- **License**: GPLv3 — all new files must include the GPL header.
- **Dependencies**: numpy, jax, h5py, dill (see `pyproject.toml` / `requirements.txt`).
- **HDF5 tests use `tempfile.TemporaryFile`** — never write temp files to disk unless testing truncation with file paths.
- **Roundtrip pattern**: most tests build a pytree, `save()` it to a temp file, `load()` it back, and assert equality via `tree_equal()`.

## Architecture notes

- The library converts arbitrary pytrees to an internal `JaxonAtom` representation, then stores atoms as HDF5 attributes/groups.
- Reference identity is preserved across save/load via object-id caching (`cached_atoms`, `loaded_objects`).
- Custom types are supported via `to_jaxon()`/`from_jaxon()` methods, `custom_marshalers`/`custom_unmarshalers` callables, or `dill` as a last resort.
- Type hint strings embedded in HDF5 group/attribute keys use the format `key:typehint`.
- Dictionary keys that are not simple strs use `key(i)`/`value(i)` pair attributes.
