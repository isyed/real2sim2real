import pytest
from real2sim2real.agentic_policy.literal_edits import bounded_keyword_vectors

SEED = '# café\nif slip > .015: raise RuntimeError("stop")\nmove(pressure_targets_n=[5., 2.5, 2.5])\nhold(pressure_targets_n=[5., 2.5, 2.5])\n'


def check(candidate):
    return bounded_keyword_vectors(
        SEED, candidate, keyword="pressure_targets_n", bounds=[(4, 6), (2, 3), (2, 3)]
    )


def test_only_existing_constant_vectors_are_editable():
    assert check(SEED)["slots"] == 2
    assert check(SEED.replace("[5., 2.5, 2.5]", "[6, 3, 3]"))["passed"]
    with pytest.raises(ValueError, match="outside permitted"):
        check(SEED.replace(".015", ".030"))
    with pytest.raises(ValueError, match="identical"):
        check(SEED.replace("[5., 2.5, 2.5]", "[6, 3, 3]", 1))


@pytest.mark.parametrize(
    "value", ["[7,3,3]", "[True,3,3]", "[5,3]", "get_targets()", '[5,float("nan"),3]']
)
def test_rejects_out_of_bounds_and_executable_expressions(value):
    with pytest.raises((ValueError, TypeError)):
        check(SEED.replace("[5., 2.5, 2.5]", value))
