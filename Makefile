# `PYTHONHASHSEED` must be set before the interpreter starts, otherwise set
# iteration order changes between runs and fuzz failures are not reproducible.
export PYTHONHASHSEED := 0

.PHONY: test fuzz

test:
	python -m pytest

fuzz:
	python -m pytest tests/test_dill.py::test_fuzzing_dill tests/test_api.py::test_fuzzing tests/test_fuzzing.py
