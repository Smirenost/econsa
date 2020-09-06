"""Calculate quantile based global sensitivity measures.

This module contains functions to calculate global sensitivity measures based on
quantiles of the output introduced by Kucherenko et al.(2019).

TODO:
    - Correct the sampling methods for exponential distribution and multivariate distribution.

"""
import numpy as np
import chaospy as cp

from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import expon


def MCS_quantile(objfun, dim, loc, scale, dist_type, N=2 ** 13, M=64, skip=0,):
    r"""Compute Monte Carlo estimators of quantile based global sensitivity measures.

    This function implements the Double loop reordering(DLR) approach described in
    Section 4.2 of [K2019]_.

    Parameters
    ----------
    objfun : callable
        Objective function to calculate the quantile-based measures. Must be broadcastable.

    dim : int
        Number of parameters of objective function.

    loc : np.ndarray or float
        The location(`loc`) keyword passed to `scipy.stats.norm`_ function to shift the
        location of "standardized" distribution. Specifically, for normal distribution
        it specifies the mean with the length of `dim`.

        .. _scipy.stats.norm: https://docs.scipy.org/doc/scipy/reference/generated/
            _scipy.stats.norm.html

    scale : np.ndarray or float
        The `scale` keyword passed to `scipy.stats.norm`_ function to adjust the scale of
        "standardized" distribution. Specifically, for normal distribution it specifies
        the covariance matrix of shape (dim, dim).

    dist_type : str
        The distribution type of input. Options are "Normal", "Exponential" and "Uniform".

    N : int
        Number of sampled points. This will later turn into the number of Monte Carlo draws.
        Accroding to [K2017]_, to preserve the uniformity properties `N` should always be
        equal to :math:`N = 2^p`, where :math:`p` is an integer. Default is :math:`2^13`.

    M : int
        Number of conditional samples. It was suggested in [K2017]_ to use as a
        "rule of thumb" :math:`M \sim \sqrt{N}`. Default is `64`.

    skip : int
        Number of values to skip of Sobol sequence. Default is `0`.

    Returns
    -------
    q1_alp : np.ndarray
        Quantile based measure. Shape has the form (len(alpha), dim).

    q2_alp : np.ndarray
        Quantile based measure. Shape has the form (len(alpha), dim).

    Q1_alp : np.ndarray
        Nomalized quantile based measure. Shape has the form (len(alpha), dim).

    Q2_alp : np.ndarray
        Nomalized quantile based measure. Shape has the form (len(alpha), dim).

    References
    ----------
    .. [K2019] S. Kucherenko, S. Song, L. Wang. Quantile based global
        sensitivity measures, Reliab. Eng. Syst. Saf. 185 (2019) 35–48.

    .. [K2017] Kucherenko S, Song S. Different numerical estimators
        for main effect global sensitivity indices. Reliab Eng Syst
        Saf 2017;165:222–38.
    """
    # range of alpha
    dalp = (0.98 - 0.02) / 30
    alp = np.arange(0.02, 0.98 + dalp, dalp)  # len(alp) = 31

    # Get quantile based measures
    q1_alp, q2_alp = _quantile_based_measures(
        objfun, dim, loc, scale, dist_type, alp, N, M, skip,
    )

    # Get nomalized quantile based measures
    Q1_alp, Q2_alp = _nomalized_quantile_based_measures(
        objfun, dim, loc, scale, dist_type, alp, N, M, skip,
    )

    return q1_alp, q2_alp, Q1_alp, Q2_alp


def _get_unconditional_sample(dim, loc, scale, dist_type, N, M, skip=0,):
    """Generate a base sample set according to joint PDF."""
    # Generate uniform distributed sample
    A = np.zeros((N, dim))
    X001 = cp.generate_samples(order=N + skip, domain=dim, rule="S").T
    X01 = X001[skip:, :dim]

    # Transform uniform draw into assigned joint PDF
    if dist_type == "Normal":
        X1 = norm.ppf(X01)
        cholesky = np.linalg.cholesky(scale)
        A = loc + cholesky.dot(X1.T).T
    elif dist_type == "Exponential":
        A = expon.ppf(X01, loc, scale)
    elif dist_type == "Uniform":
        A = uniform.ppf(X01, loc, scale)
    else:
        raise NotImplementedError

    return A


def _get_conditional_sample(dim, loc, scale, dist_type, N, M, skip,):
    """Generate a conditional sample set from the base sample set."""
    A = _get_unconditional_sample(dim, loc, scale, dist_type, N, M, skip,)
    B = A[:M]
    # conditional sample matrix C with shape of (M, dim, N, dim)
    C = np.array(
        [[np.zeros((N, dim)) for x in range(dim)] for z in range(M)], dtype=np.float64
    )

    for i in range(dim):
        for j in range(M):
            C[j, i] = A
            C[j, i, :, i] = B[j, i]

    return C


def _unconditional_q_Y(objfun, dim, loc, scale, dist_type, alp, N, M, skip,):
    """Calculate quantiles of outputs with base sample set as inputs."""
    A = _get_unconditional_sample(dim, loc, scale, dist_type, N, M, skip,)

    # Equation 26 & 23
    Y1 = objfun(A)  # values of outputs
    y1 = np.sort(Y1)  # reorder in ascending order
    q_index = (np.floor(alp * N) - 1).astype(int)
    qy_alp1 = y1[q_index]  # quantiles corresponding to alpha

    return qy_alp1


def _conditional_q_Y(objfun, dim, loc, scale, dist_type, alp, N, M, skip,):
    """Calculate quantiles of outputs with conditional sample set as inputs."""
    C = _get_conditional_sample(
        dim, loc, scale, dist_type, N, M, skip,
    )  # shape(M, dim, N, dim)

    # initialize values of conditional outputs.
    Y2 = np.array(
        [[np.zeros((N, 1)) for x in range(dim)] for z in range(M)], dtype=np.float64
    )  # shape(N, dim, N, 1)
    y2 = np.array(
        [[np.zeros((N, 1)) for x in range(dim)] for z in range(M)], dtype=np.float64
    )

    # initialize quantile of conditional outputs.
    qy_alp2 = np.array(
        [[np.zeros((len(alp), M)) for x in range(dim)] for z in range(1)],
        dtype=np.float64,
    )  # shape(1, dim, len(alp), M)

    # Equation 26 & 23
    for i in range(dim):
        for j in range(M):
            # values of conditional outputs
            Y2[j, i] = np.vstack(objfun(C[j, i]))
            Y2[j, i].sort(axis=0)
            y2[j, i] = Y2[j, i]  # reorder in ascending order
            # conditioanl q_Y(alp)
            for pp in range(len(alp)):
                qy_alp2[0, i, pp, j] = y2[j, i][
                     (np.floor(alp[pp] * N) - 1).astype(int) 
                ]  # quantiles corresponding to alpha
    return qy_alp2


def _quantile_based_measures(objfun, dim, loc, scale, dist_type, alp, N, M, skip,):
    """Compute MC/QMC estimators of quantile based measures."""
    qy_alp1 = _unconditional_q_Y(objfun, dim, loc, scale, dist_type, alp, N, M, skip,)
    qy_alp2 = _conditional_q_Y(objfun, dim, loc, scale, dist_type, alp, N, M, skip,)

    # initialization
    q1_alp = np.zeros((len(alp), dim))
    q2_alp = np.zeros((len(alp), dim))
    delt = np.array(
        [[np.zeros((1, M)) for x in range(dim)] for z in range(1)], dtype=np.float64
    )

    # Equation 27 & 28
    for i in range(dim):
        for pp in range(len(alp)):
            delt[0, i] = qy_alp2[0, i, pp, :] - qy_alp1[pp]  # delt
            q1_alp[pp, i] = np.mean(np.absolute(delt[0, i]))  # |delt|
            q2_alp[pp, i] = np.mean(delt[0, i] ** 2)  # (delt)^2

    return q1_alp, q2_alp


def _nomalized_quantile_based_measures(
    objfun, dim, loc, scale, dist_type, alp, N, M, skip,
):
    """Compute MC/QMC estimators of nomalized quantile based measures."""
    q1_alp, q2_alp = _quantile_based_measures(
        objfun, dim, loc, scale, dist_type, alp, N, M, skip,
    )

    # initialize quantile measures arrays.
    q1 = np.zeros(len(alp))
    q2 = np.zeros(len(alp))
    Q1_alp = np.zeros((len(alp), dim))
    Q2_alp = np.zeros((len(alp), dim))

    # Equation 13 & 14
    for pp in range(len(alp)):
        q1[pp] = np.sum(q1_alp[pp, :])
        q2[pp] = np.sum(q2_alp[pp, :])
        for i in range(dim):
            Q1_alp[pp, i] = q1_alp[pp, i] / q1[pp]
            Q2_alp[pp, i] = q2_alp[pp, i] / q2[pp]

    return Q1_alp, Q2_alp
