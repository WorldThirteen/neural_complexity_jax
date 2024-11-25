import pytest
from jax import jit, config, numpy as jnp

config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)
import numpy as np
from neural_complexity import (
    calc_approximate_complexity,
    numpy_mt19937_gaussian_generator,
)

test_cases = [
    {
        "csv_file": "tests/data/polyworld_brain_activity_1.csv",
        "expected_complexity_all_points": 5.848132,
        "expected_complexity_simplified": 0.999624,
    },
    {
        "csv_file": "tests/data/polyworld_brain_activity_2.csv",
        "expected_complexity_all_points": 7.923551,
        "expected_complexity_simplified": 0.561005,
    },
]

jcalc_approximate_complexity = jit(calc_approximate_complexity, static_argnums=(1, 2))


@pytest.mark.parametrize("test_case", test_cases)
def test_calc_approximate_complexity(test_case):
    """Test calc_approximate_complexity with multiple input files and expected results."""

    # Load the input matrix from the CSV file
    input_matrix = np.loadtxt(test_case["csv_file"], delimiter=",")
    input_matrix = jnp.array(input_matrix)

    # Using same algorithm & seed for near reproducibility while being slower
    gauss_generator = numpy_mt19937_gaussian_generator(42)

    # Test complexity with all points
    computed_complexity_all_points = jcalc_approximate_complexity(
        input_matrix, 0, gauss_generator
    )
    assert jnp.isclose(
        computed_complexity_all_points,
        test_case["expected_complexity_all_points"],
        rtol=0.1,
    ), f"calc_approximate_complexity mismatch for {test_case['csv_file']} (all points): {computed_complexity_all_points} != {test_case['expected_complexity_all_points']}"

    # Test complexity with simplified calculation
    computed_complexity_simplified = jcalc_approximate_complexity(
        input_matrix, 1, gauss_generator
    )
    assert jnp.isclose(
        computed_complexity_simplified,
        test_case["expected_complexity_simplified"],
        rtol=0.1,
    ), f"calc_approximate_complexity mismatch for {test_case['csv_file']} (simplified): {computed_complexity_simplified} != {test_case['expected_complexity_simplified']}"
