# This Python implementation is an adaptation of the original C++ implementation by L.Yaeger for calculating complexity metrics.
# The approach includes Gaussianization, determinant calculation using LU decomposition, and integration estimation.
# Original C++ implementation: https://github.com/polyworld/polyworld
# Licensed under the APSL License.
# Modified by Mykhailo Zachepylo in 2024.

from jax import (
    jax,
    random,
    numpy as jnp,
)
from jax.scipy.linalg import lu
import numpy as np
from itertools import combinations

DEFAULT_SEED = 42
defaultKey = random.PRNGKey(DEFAULT_SEED)


# NumPy-based generator with MT19937
def numpy_mt19937_gaussian_generator(seed=DEFAULT_SEED):
    rng = np.random.default_rng(np.random.MT19937(seed))

    def gaussian_generator(shape):
        return rng.normal(size=shape)

    return gaussian_generator


# JAX-based generator
def jax_gaussian_generator(seed=DEFAULT_SEED):
    key = random.PRNGKey(seed)

    def gaussian_generator(shape):
        return random.normal(key, shape=shape)

    return gaussian_generator


def gaussianize_matrix(matrix: jnp.ndarray, gaussian_generator) -> jnp.ndarray:
    """
    Applies Gaussianization to each column of the input matrix.

    Each column is ranked, and these ranks are then used to map the data into a sorted Gaussian distribution.
    This process ensures that each column is transformed to match a standard normal distribution while retaining
    the rank structure of the original data. The Gaussian values are generated for the entire matrix at once.
    """
    r, c = matrix.shape

    # Generate 2D Gaussian random values
    gauss = gaussian_generator((r, c))
    gauss_sorted = jnp.sort(gauss, axis=0)  # Sort each column

    # Compute ranks for each column
    ranks = jnp.argsort(jnp.argsort(matrix, axis=0), axis=0)

    # Map ranks to Gaussian values
    gaussianized_matrix = gauss_sorted[ranks, jnp.arange(c)]
    return gaussianized_matrix


def determinant(matrix: jnp.ndarray) -> jnp.float64:
    """
    Computes the determinant of a matrix using LU decomposition.

    LU decomposition splits the matrix into lower and upper triangular matrices (L and U).
    The determinant is the product of the diagonal elements of U, adjusted by the permutation matrix.
    This approach provides a numerically stable way of computing the determinant, particularly for large matrices.
    """
    P, L, U = lu(matrix)
    det_U = jnp.prod(jnp.diag(U))
    det_P = jnp.linalg.det(P)
    return jnp.abs(det_P * det_U)


def calc_I(matrix: jnp.ndarray) -> jnp.float64:
    """
    Calculates integration I(X) from the covariance matrix and its determinant.

    This function quantifies the degree of integration within the system by comparing the determinant
    of the covariance matrix with the product of its diagonal elements. Higher values indicate more
    integrated relationships among the variables in the system.

    Integration:

        I(X) = ∑[ H(Xᵢ) ] - H(X)

    Where:
        - H(X) is the entropy of the entire system.
        - H(Xᵢ) is the entropy of each individual variable.

    Note: This calculation uses log2, consistent with the original implementation's use of information theory metrics.
    """
    determinant_value = determinant(matrix)
    diag_log_sum = jnp.sum(jnp.log2(jnp.diag(matrix)))
    det_log = jnp.log2(determinant_value)

    return 0.5 * (diag_log_sum - det_log)


def n_choose_k_le_s(n: int, k: int, s: int) -> bool:
    """
    Checks if n choose k is <= s without computing the full value.

    This function efficiently determines if the combination of n choose k exceeds a given threshold s.
    It stops the computation as soon as the product exceeds s, which helps in avoiding excessive calculations
    when only a comparison is needed. This is useful for determining if exhaustive enumeration is feasible.

    Note: This is only accurate up to about <n choose k> < 10^13, which should be more than adequate for most purposes.
    """
    product = 1.0
    for i in range(1, k + 1):
        product *= (n - (k - i)) / i
        if product > s:
            return False
    return True


def calc_C_k(
    cov_matrix: jnp.ndarray, I_n: float, k: int, num_samples: int = 1000
) -> float:
    """
    Calculates Cₖ using a sampled approach for large k to avoid exhaustive computation.

    For larger values of k, calculating all possible combinations is computationally infeasible.
    This function approximates Cₖ by using a random sample of subsets, which provides a trade-off
    between computational efficiency and accuracy. Edge cases (k = 1 or k = n - 1) are calculated exactly.

    Complexity formula (TSE complexity):

        Cₙ(X) = ∑[ (k/n) I(X) - ⟨I(Xₖ)⟩ ]

    Where:
        - I(X) is the integration of the full system.
        - ⟨I(Xₖ)⟩ is the average integration over all subsets of size k.
    """
    n = cov_matrix.shape[0]
    LI_k = (k / n) * I_n

    # Use exact calculation for edge cases
    if k == n - 1 or k == 1 or n_choose_k_le_s(n, k, num_samples):
        return calc_C_k_exact(cov_matrix, I_n, k)

    @jax.jit
    def sample_calc_I_k(key):
        indices = jnp.sort(random.choice(key, n, shape=(k,), replace=False))
        sub_cov_matrix = cov_matrix[jnp.ix_(indices, indices)]
        return calc_I(sub_cov_matrix)

    keys = random.split(defaultKey, num_samples)
    EI_k = jnp.mean(jax.vmap(sample_calc_I_k)(keys))

    return LI_k - EI_k


def calc_C_k_exact(cov_matrix: jnp.ndarray, I_n: float, k: int) -> float:
    """
    Calculates Cₖ exactly by iterating over all subsets of size k.

    This function generates all possible combinations of size k from the set of variables and
    computes the corresponding integration values for each subset. This exhaustive approach ensures
    maximum accuracy but can be computationally expensive for large k and n.

    Uses all subsets of size k from a set of size n to calculate Cₖ(X) = (k/N) * I(X) - ⟨I(Xₖ)⟩.
    """
    n = cov_matrix.shape[0]
    LI_k = (k / n) * I_n

    # Generate all possible index combinations
    indices = jnp.array([jnp.array(comb) for comb in combinations(range(n), k)])

    @jax.jit
    def calc_I_for_combination(idx):
        sub_cov_matrix = cov_matrix[jnp.ix_(idx, idx)]
        return calc_I(sub_cov_matrix)

    sum_I_k = jnp.sum(jax.vmap(calc_I_for_combination)(indices))
    num_combinations = indices.shape[0]

    EI_k = sum_I_k / num_combinations
    return LI_k - EI_k


def calc_approximate_complexity(
    matrix: jnp.ndarray, num_points=1, gaussian_generator=jax_gaussian_generator()
) -> float:
    """
    Calculates the approximate full complexity C(X).

    This function computes the overall complexity of the system by integrating over multiple subsets of the dataset.
    It begins by adding noise to regularize the covariance matrix, then Gaussianize each column to standardize the data.
    Complexity is calculated by considering how integrated different subsets of the system are, with more integrated
    systems indicating higher complexity.

        Cₙ(X) = ∑[ ⟨H(Xₖₙ)⟩ - (k/n) H(X) ]

    Where:
        - H(X) is the entropy of the entire system.
        - ⟨H(Xₖₙ)⟩ is the average entropy over all n!/(k!(n - k)!) combinations of k variables.

    Entropy calculation for a system with covariance matrix COV:

        H(X) = 1/2 ln((2πe)ⁿ |COV|)
    """
    if matrix is None or matrix.size == 0:
        raise ValueError("Input matrix is invalid.")

    # Add noise to the data (helps to regularize the covariance matrix)
    noise_scale = 0.00001
    noise = random.normal(defaultKey, matrix.shape) * noise_scale
    matrix += noise

    # Gaussianize the data to make it more normally distributed
    matrix = gaussianize_matrix(matrix, gaussian_generator)

    # Calculate the covariance matrix of the Gaussianized data
    cov_matrix = jnp.cov(matrix, rowvar=False)

    # Calculate the integration of the full matrix I(X)
    I_n = calc_I(cov_matrix)

    n = cov_matrix.shape[0]
    complexity = 0

    if num_points <= 0 or num_points >= n:
        # Use all points if num_points is invalid
        num_points = n - 1

    if num_points == 1:
        # Simplified case: use only the k = n - 1 term
        complexity = calc_C_k_exact(cov_matrix, I_n, n - 1)
    else:
        delta_k = (n - 2) / (num_points - 1)
        k_prev = 1
        c_prev = I_n / n
        complexity += 0.5 * (c_prev)  # Initial triangular area

        float_k = 1.0
        for i in range(2, num_points):
            float_k += delta_k
            k = int(round(float_k))
            c_k = calc_C_k(cov_matrix, I_n, k)
            dk = k - k_prev
            complexity += 0.5 * dk * (c_k + c_prev)
            k_prev = k
            c_prev = c_k

        # Final term k = n-1
        c_k = calc_C_k(cov_matrix, I_n, n - 1)
        dk = (n - 1) - k_prev
        complexity += 0.5 * dk * (c_k + c_prev)

        # Normalize by dividing by n
        complexity /= n

    return complexity
