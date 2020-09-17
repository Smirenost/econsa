"""Test for quantile based global sensitivity measures.

Analytical values of linear model with normally distributed variables
are used as benchmarks for verification of numerical estimates.
"""
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import norm

from econsa.quantile_measures import mcs_quantile


@pytest.fixture
def first_example_fixture():
    """First example test case."""

    def func1(args):
        """Function for test 1 from Kucherenko et al. 2019."""
        result = np.sum(args, axis=1)
        return result

    miu_1 = np.array([1, 3, 5, 7])
    cov_1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 2.25, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 6.25],
        ],
    )
    dim_1 = len(miu_1)

    # range of alpha
    dalp = (0.98 - 0.02) / 30
    alp = np.arange(0.02, 0.98 + dalp, dalp)  # len(alp) = 31

    # inverse error function
    phi_inv = norm.ppf(alp)

    # q_2: PDF of the out put Y(Eq.30)
    expect_q2 = []

    for a in range(len(alp)):
        q2_a = []
        for i in range(dim_1):
            q2_i = (
                cov_1[i, i]
                + phi_inv[a] ** 2
                * (
                    np.sqrt(np.trace(cov_1))
                    - np.sqrt(sum(cov_1[j, j] for j in range(dim_1) if j != i))
                )
                ** 2
            )
            q2_a.append(q2_i)
        expect_q2.append(q2_a)

    expect_q2 = np.vstack(expect_q2).reshape((len(alp), dim_1))

    # warning: this works only in this case
    expect_q1 = np.sqrt(expect_q2)

    # Q_2: normalized quantile based sensitivity measure 2.(Eq.14)
    expect_normalized_q2 = []

    for a in range(len(alp)):
        normalized_q2_a = []
        for i in range(dim_1):
            normalized_q2_i = expect_q2[a, i] / sum(expect_q2[a])
            normalized_q2_a.append(normalized_q2_i)
        expect_normalized_q2.append(normalized_q2_a)

    expect_normalized_q2 = np.hstack(expect_normalized_q2).reshape((len(alp), dim_1))

    # Q_1: normalized quantile based sensitivity measure
    expect_normalized_q1 = []

    for a in range(len(alp)):
        normalized_q2_a = []
        for i in range(dim_1):
            normalized_q1_i = expect_q1[a, i] / sum(expect_q1[a])
            normalized_q2_a.append(normalized_q1_i)
        expect_normalized_q1.append(normalized_q2_a)

    expect_normalized_q1 = np.hstack(expect_normalized_q1).reshape((len(alp), dim_1))

    # Combine results
    quantile_measures_expected = (expect_normalized_q1, expect_normalized_q2)

    out = {
        "func": func1,
        "n_params": dim_1,
        "loc": miu_1,
        "scale": cov_1,
        "dist_type": "Normal",
        "n_draws": 2 ** 13,
        "quantile_measures_expected": quantile_measures_expected,
    }

    return out


def test_quantile_measures_first_example(first_example_fixture):
    quantile_measures_expected = first_example_fixture["quantile_measures_expected"]
    func = first_example_fixture["func"]
    n_params = first_example_fixture["n_params"]
    loc = first_example_fixture["loc"]
    scale = first_example_fixture["scale"]
    dist_type = first_example_fixture["dist_type"]
    n_draws = first_example_fixture["n_draws"]

    quantile_measures = mcs_quantile(
        func=func,
        n_params=n_params,
        loc=loc,
        scale=scale,
        dist_type=dist_type,
        n_draws=n_draws,
    )

    assert_almost_equal(quantile_measures[2:], quantile_measures_expected, decimal=2)
