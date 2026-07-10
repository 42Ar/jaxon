import dill
from .testing.fuzz import fuzz_tree_generator
from .testing.tree_equal import assert_tree_equal


def test_fuzzing_dill():
    for pytree in fuzz_tree_generator(1000):
        loaded = dill.loads(dill.dumps(pytree))
        assert_tree_equal(pytree, loaded)
