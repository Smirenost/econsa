"""Tests for the sampling.

This module contains all tests for the sampling setup.

"""
import numpy as np
import pytest
from numpy.random import default_rng

from econsa.sampling import cond_mvn
from econsa.tests.wrapper_r import r_cond_mvn

rng = default_rng()


def get_strategies(name):
    n = rng.integers(low=4, high=21)
    mean = rng.integers(low=-2, high=2, size=n)
    dependent_n = rng.integers(low=1, high=n - 2)
    dependent_ind = rng.choice(range(0, n), replace=False, size=dependent_n)

    if name == "cond_mvn":
        cov = rng.standard_normal(size=(n, n))
        cov = cov @ cov.T
        given_ind = [x for x in range(0, n) if x not in dependent_ind]
        given_value = rng.uniform(low=-2, high=2, size=len(given_ind))
    elif name == "cond_mvn_exception_given":
        cov = rng.standard_normal(size=(n, n))
        cov = cov @ cov.T
        given_ind = (
            [x for x in range(0, n) if x not in dependent_ind] if n % 3 == 0 else None
        )
        given_value = (
            rng.uniform(low=-2, high=2, size=n - dependent_n + 1)
            if n % 2 == 0
            else None
        )
    elif name == "test_cond_mvn_exception_cov":
        cov = rng.standard_normal(size=(n, n)) if n % 3 == 0 else np.diagflat([-1] * n)
        cov = cov @ cov.T if n % 2 == 0 else cov
        given_ind = [x for x in range(0, n) if x not in dependent_ind]
        given_value = rng.uniform(low=-2, high=2, size=len(given_ind))
    else:
        raise NotImplementedError

    strategy = (mean, cov, dependent_ind, given_ind, given_value)

    return strategy


def test_cond_mvn():
    """Test cond_mvn against the original from R package cond_mvnorm."""
    request = get_strategies("cond_mvn")

    r_cond_mean, r_cond_cov = r_cond_mvn(*request)
    cond_mean, cond_cov = cond_mvn(*request)

    np.testing.assert_allclose(cond_mean, r_cond_mean)
    np.testing.assert_allclose(cond_cov, r_cond_cov)


def test_cond_mvn_exception_given():
    """Test cond_mvn raises exceptions when invalid `given_ind` or `given_value` is passed."""
    mean, cov, dependent_ind, given_ind, given_value = get_strategies(
        "cond_mvn_exception_given",
    )

    n = cov.shape[0]
    if n % 3 != 0:
        # Valid case: only `given_ind` is empty or both `given_ind` and `given_value` are empty
        cond_mvn(mean, cov, dependent_ind, given_ind, given_value)
    else:
        # `given_value` is empty or does not align with `given_ind`
        with pytest.raises(ValueError) as e:
            cond_mvn(mean, cov, dependent_ind, given_ind, given_value)
        assert "lengths of given_value and given_ind must be the same" in str(e.value)


def test_cond_mvn_exception_cov():
    """Test cond_mvn raises exceptions when invalid `cov` is passed."""
    mean, cov, dependent_ind, given_ind, given_value = get_strategies(
        "test_cond_mvn_exception_cov",
    )

    n = cov.shape[0]

    if n % 3 != 0 and n % 2 != 0:
        # `cov` is negative definite matrix
        with pytest.raises(ValueError) as e:
            cond_mvn(mean, cov, dependent_ind, given_ind, given_value)
        assert "cov is not positive-definite" in str(e.value)
    elif n % 2 != 0:
        # `cov` is not symmetric
        with pytest.raises(ValueError) as e:
            cond_mvn(mean, cov, dependent_ind, given_ind, given_value)
        assert "cov is not a symmetric matrix" in str(e.value)
    else:
        cond_mvn(mean, cov, dependent_ind, given_ind, given_value)
