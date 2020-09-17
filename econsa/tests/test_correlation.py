"""Test for correlation transformation."""
import chaospy as cp
import numpy as np
import pytest
from scipy.stats import norm

from econsa.copula import _cov2corr
from econsa.correlation import _find_positive_definite
from econsa.correlation import gc_correlation


def get_strategies(name):
    dim = np.random.randint(2, 10)
    means = np.random.uniform(-100, 100, dim)

    if name == "test_gc_correlation_functioning":
        # List of distributions to draw from.
        distributions = [
            cp.Exponential,
            cp.Gilbrat,
            cp.HyperbolicSecant,
            cp.Laplace,
            cp.LogNormal,
            cp.LogUniform,
            cp.LogWeibull,
            cp.Logistic,
            cp.Maxwell,
            cp.Normal,
            cp.Rayleigh,
            cp.Uniform,
            cp.Wigner,
        ]
        marginals = list()
        for mean in means:
            dist = distributions[np.random.choice(len(distributions))](mean)
            marginals.append(dist)

        cov = np.random.uniform(-1, 1, size=(dim, dim))
        cov = cov @ cov.T
        cov = _find_positive_definite(cov)
        # The rounding is necessary to prevent ValueError("corr must be between 0 and 1")
        corr = _cov2corr(cov).round(8)
    elif (
        name == "test_gc_correlation_2d" or name == "test_gc_correlation_2d_force_calc"
    ):
        dim = 2
        means = np.random.uniform(-100, 100, dim)
        distributions = [
            cp.Normal,
            cp.Uniform,
            cp.Exponential,
            cp.Rayleigh,
            cp.LogWeibull,
        ]
        marginals = [cp.Normal(means[0])]
        dist2 = distributions[np.random.choice(len(distributions))](means[1])
        marginals.append(dist2)

        corr = np.identity(2)
        corr[0, 1] = corr[1, 0] = np.random.uniform(-0.75, 0.75)

    elif name == "test_gc_correlation_exception_marginals":
        marginals = list()
        for i in range(dim):
            marginals.append(norm())

        corr = np.random.uniform(-1, 1, size=(dim, dim))
    elif name == "test_gc_correlation_exception_corr_symmetric":
        distributions = [cp.Normal, cp.Uniform, cp.LogNormal]
        marginals = list()
        for mean in means:
            dist = distributions[np.random.choice(len(distributions))](mean)
            marginals.append(dist)

        corr = np.random.uniform(-1, 1, size=(dim, dim))
    else:
        raise NotImplementedError

    strategy = (marginals, corr)
    return strategy


def test_gc_correlation_functioning():
    """Test the function runs successfully."""
    marginals, corr = get_strategies("test_gc_correlation_functioning")
    corr_transformed = gc_correlation(marginals, corr)
    cp.Nataf(cp.J(*marginals), corr_transformed)

    return "the function ended without error"


def test_gc_correlation_2d():
    """Test the results from the paper are accurate."""
    marginals, corr_desired = get_strategies("test_gc_correlation_2d")
    rtol, atol = 0.01, 0.01

    corr_transformed = gc_correlation(marginals, corr_desired)
    copula = cp.Nataf(cp.J(*marginals), corr_transformed)
    corr_copula = np.corrcoef(copula.sample(10000000))

    np.testing.assert_allclose(corr_desired, corr_copula, rtol, atol)


def test_gc_correlation_2d_force_calc():
    """Test for low dimensional special cases the results are accurate."""
    marginals, corr_desired = get_strategies("test_gc_correlation_2d_force_calc")

    rtol, atol = 0.01, 0.01
    precision = atol + rtol * abs(corr_desired[0, 1])

    candidates = dict()
    candidates["corr"], candidates["stat"] = list(), list()

    is_success = False
    for order in [5, 10, 15, 20]:
        if is_success:
            break

        kwargs = dict()
        kwargs["order"] = order

        corr_transformed = gc_correlation(
            marginals,
            corr_desired,
            **kwargs,
            force_calc=True,
        )
        copula = cp.Nataf(cp.J(*marginals), corr_transformed)
        corr_copula = np.corrcoef(copula.sample(10000000))
        candidates["stat"].append(np.abs(corr_desired[0, 1] - corr_copula[0, 1]))
        candidates["corr"].append(corr_copula)

        is_success = candidates["stat"][-1] < precision

    corr_copula = candidates["corr"][np.argmin(candidates["stat"])]
    np.testing.assert_allclose(corr_desired, corr_copula, rtol, atol)


def test_gc_correlation_exception_marginals():
    marginals, corr = get_strategies("test_gc_correlation_exception_marginals")

    with pytest.raises(NotImplementedError) as e:
        gc_correlation(marginals, corr)
    assert "marginals must be chaospy distributions" in str(e.value)


def test_gc_correlation_exception_corr_symmetric():
    marginals, corr = get_strategies("test_gc_correlation_exception_corr_symmetric")

    with pytest.raises(ValueError) as e:
        gc_correlation(marginals, corr)
    assert "corr is not a symmetric matrix" in str(e.value)
