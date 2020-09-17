"""Conditional sampling from Gaussian copula.

This module contains functions to sample random input parameters from a Gaussian copula.
"""
import numpy as np
from scipy.stats import multivariate_normal as multivariate_norm
from scipy.stats import norm

from econsa.sampling import cond_mvn


def cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u, size=1):
    r"""Conditional sampling from Gaussian copula.

    This function provides the probability distribution of conditional sample
    drawn from a Gaussian copula, given covariance matrix and a uniform random vector.

    Parameters
    ----------
    cov : array_like
        Covariance matrix of the desired sample.

    dependent_ind : int or array_like
        The indices of dependent variables.

    given_ind : array_like
        The indices of independent variables.

    given_value_u : array_like
        The given random vector (:math:`u`) that is uniformly distributed between 0 and 1.

    size : int
        Number of draws from the conditional distribution. (default value is `1`)

    Returns
    -------
    cond_quan : numpy.ndarray
        The conditional sample (:math:`G(u)`) that is between 0 and 1,
        and has the same length as ``dependent_ind``.

    Examples
    --------
    >>> np.random.seed(123)
    >>> cov = np.array([[ 3.290887,  0.465004, -3.411841],
    ...                 [ 0.465004,  3.962172, -0.574745],
    ...                 [-3.411841, -0.574745,  4.063252]])
    >>> dependent_ind = 2
    >>> given_ind = [0, 1]
    >>> given_value_u = [0.0596779, 0.39804426]
    >>> condi_value_u = cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u)
    >>> np.testing.assert_almost_equal(condi_value_u[0], 0.856504, decimal=6)
    """
    given_value_u = np.atleast_1d(given_value_u)

    # Check `given_value_u` are between 0 and 1
    if not np.all((given_value_u >= 0) & (given_value_u <= 1)):
        raise ValueError("given_value_u must be between 0 and 1")

    # F^{−1}(u)
    given_value_y = norm().ppf(given_value_u)

    mean = np.zeros(cov.shape[0])
    cond_mean, cond_cov = cond_mvn(
        mean,
        _cov2corr(cov),
        dependent_ind,
        given_ind,
        given_value_y,
    )

    # C(u, Sigma)
    cond_dist = multivariate_norm(cond_mean, cond_cov)
    cond_draw = np.atleast_1d(cond_dist.rvs(size=size))
    cond_quan = np.atleast_1d(norm.cdf(cond_draw))

    return cond_quan


def _cov2corr(cov, return_std=False):
    r"""Convert covariance matrix to correlation matrix.

    This function does not convert subclasses of `ndarray`s.
    This requires that division is defined element-wise.
    `np.ma.array` and `np.matrix` are allowed.

    Parameters
    ----------
    cov: array_like
         Covariance matrix with dimensions :math:`N\times N`.

    return_std: bool, optional
         If True then the standard deviation is also returned. (default value is `False`)

    Returns
    -------
    corr: ndarray
        Correlation matrix with dimensions :math:`N\times N`.

    std_: ndarray
        Standard deviation with dimensions :math:`1\times N`.
    """
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr
