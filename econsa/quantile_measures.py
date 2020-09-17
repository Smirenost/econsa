"""Calculate quantile based global sensitivity measures.

This module contains functions to calculate global sensitivity measures based on
quantiles of the output introduced by Kucherenko et al.(2019).

TODO:
    - Correct the sampling methods for exponential distribution and multivariate distribution.

"""
import chaospy as cp
import numpy as np
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import uniform


def mcs_quantile(
    func,
    n_params,
    loc,
    scale,
    dist_type,
    n_draws=2 ** 13,
    m=64,
    skip=0,
):
    r"""Compute Monte Carlo estimators of quantile based global sensitivity measures.

    This function implements the Double loop reordering(DLR) approach described in
    Section 4.2 of [K2019]_.

    Parameters
    ----------
    func : callable
        Objective function to calculate the quantile-based measures. Must be broadcastable.

    n_params : int
        Number of parameters of objective function.

    loc : np.ndarray or float
        The location(`loc`) keyword passed to `scipy.stats.norm`_ function to shift the
        location of "standardized" distribution. Specifically, for normal distribution
        it specifies the mean with the length of `n_params`.

        .. _scipy.stats.norm: https://docs.scipy.org/doc/scipy/reference/generated/
            _scipy.stats.norm.html

    scale : np.ndarray or float
        The `scale` keyword passed to `scipy.stats.norm`_ function to adjust the scale of
        "standardized" distribution. Specifically, for normal distribution it specifies
        the covariance matrix of shape (n_params, n_params).

    dist_type : str
        The distribution type of input. Options are "Normal", "Exponential" and "Uniform".

    n_draws : int
        Number of sampled points. This will later turn into the number of Monte Carlo draws.
        Accroding to [K2017]_, to preserve the uniformity properties `n_draws` should always be
        equal to :math:`n_draws = 2^p`, where :math:`p` is an integer. Default is :math:`2^13`.

    m : int
        Number of conditional samples. It was suggested in [K2017]_ to use as a
        "rule of thumb" :math:`m \sim \sqrt{n_draws}`. Default is `64`.

    skip : int
        Number of values to skip of Sobol sequence. Default is `0`.

    Returns
    -------
    q1_alp : np.ndarray
        Quantile based measure. Shape has the form (len(alp), n_params).

    q2_alp : np.ndarray
        Quantile based measure. Shape has the form (len(alp), n_params).

    nomalized_q1_alp : np.ndarray
        Nomalized quantile based measure. Shape has the form (len(alp), n_params).

    nomalized_q2_alp : np.ndarray
        Nomalized quantile based measure. Shape has the form (len(alp), n_params).

    References
    ----------
    .. [K2019] Kucherenko, S., Song, S., & Wang, L. (2019).
        Quantile based global sensitivity measures.
        Reliability Engineering & System Safety, 185, 35-48.

    .. [K2017] Kucherenko, S. & Song, S. (2017).
        Different numerical estimators for main effect global
        sensitivity indices. Reliability Engineering & System
        Safety, 165, 222-238.
    """
    # range of alpha
    dalp = (0.98 - 0.02) / 30
    alp = np.arange(0.02, 0.98 + dalp, dalp)  # len(alp) = 31

    # Get quantile based measures
    q1_alp, q2_alp = _quantile_based_measures(
        func,
        n_params,
        loc,
        scale,
        dist_type,
        alp,
        n_draws,
        m,
        skip,
    )

    # Get nomalized quantile based measures
    nomalized_q1_alp, nomalized_q2_alp = _nomalized_quantile_based_measures(
        func,
        n_params,
        loc,
        scale,
        dist_type,
        alp,
        n_draws,
        m,
        skip,
    )

    return q1_alp, q2_alp, nomalized_q1_alp, nomalized_q2_alp


def _get_unconditional_sample(
    n_params,
    loc,
    scale,
    dist_type,
    n_draws,
    m,
    skip=0,
):
    """Generate a base sample set according to joint PDF."""
    # Generate uniform distributed sample
    u_1 = cp.generate_samples(order=n_draws + skip, domain=n_params, rule="S").T
    u_2 = u_1[skip:, :n_params]
    unconditional_sample = np.zeros((n_draws, n_params))

    # Transform uniform draw into assigned joint PDF
    if dist_type == "Normal":
        z = norm.ppf(u_2)
        cholesky = np.linalg.cholesky(scale)
        unconditional_sample = loc + cholesky.dot(z.T).T
    elif dist_type == "Exponential":
        unconditional_sample = expon.ppf(u_2, loc, scale)
    elif dist_type == "Uniform":
        unconditional_sample = uniform.ppf(u_2, loc, scale)
    else:
        raise NotImplementedError

    return unconditional_sample


def _get_conditional_sample(
    n_params,
    loc,
    scale,
    dist_type,
    n_draws,
    m,
    skip,
):
    """Generate a conditional sample set from the base sample set."""
    unconditional_sample = _get_unconditional_sample(
        n_params,
        loc,
        scale,
        dist_type,
        n_draws,
        m,
        skip,
    )
    conditional_bin = unconditional_sample[:m]
    # conditional sample matrix with shape of (m, n_params, n_draws, n_params)
    conditional_sample = np.array(
        [[np.zeros((n_draws, n_params)) for x in range(n_params)] for z in range(m)],
        dtype=np.float64,
    )

    for i in range(n_params):
        for j in range(m):
            conditional_sample[j, i] = unconditional_sample
            conditional_sample[j, i, :, i] = conditional_bin[j, i]

    return conditional_sample


def _unconditional_q_y(
    func,
    n_params,
    loc,
    scale,
    dist_type,
    alp,
    n_draws,
    m,
    skip,
):
    """Calculate quantiles of outputs with unconditional sample set as inputs."""
    unconditional_sample = _get_unconditional_sample(
        n_params,
        loc,
        scale,
        dist_type,
        n_draws,
        m,
        skip,
    )

    # Equation 26 & 23
    y_1 = func(unconditional_sample)  # values of outputs
    y_1_asc = np.sort(y_1)  # reorder in ascending order
    q_index = (np.floor(alp * n_draws) - 1).astype(int)
    qy_alp1 = y_1_asc[q_index]  # quantiles corresponding to alpha

    return qy_alp1


def _conditional_q_y(
    func,
    n_params,
    loc,
    scale,
    dist_type,
    alp,
    n_draws,
    m,
    skip,
):
    """Calculate quantiles of outputs with conditional sample set as inputs."""
    conditional_sample = _get_conditional_sample(
        n_params,
        loc,
        scale,
        dist_type,
        n_draws,
        m,
        skip,
    )  # shape(m, n_params, n_draws, n_params)

    # initialize values of conditional outputs.
    y_2 = np.array(
        [[np.zeros((n_draws, 1)) for x in range(n_params)] for z in range(m)],
        dtype=np.float64,
    )  # shape(n_draws, n_params, n_draws, 1)
    y_2_asc = np.array(
        [[np.zeros((n_draws, 1)) for x in range(n_params)] for z in range(m)],
        dtype=np.float64,
    )
    # initialize quantile of conditional outputs.
    qy_alp2 = np.array(
        [[np.zeros((len(alp), m)) for x in range(n_params)] for z in range(1)],
        dtype=np.float64,
    )  # shape(1, n_params, len(alp), m)

    # Equation 26 & 23
    for i in range(n_params):
        for j in range(m):
            # values of conditional outputs
            y_2[j, i] = np.vstack(func(conditional_sample[j, i]))
            y_2[j, i].sort(axis=0)
            y_2_asc[j, i] = y_2[j, i]  # reorder in ascending order
            # conditioanl q_y(alp)
            for pp in range(len(alp)):
                qy_alp2[0, i, pp, j] = y_2_asc[j, i][
                    (np.floor(alp[pp] * n_draws) - 1).astype(int)
                ]  # quantiles corresponding to alpha
    return qy_alp2


def _quantile_based_measures(
    func,
    n_params,
    loc,
    scale,
    dist_type,
    alp,
    n_draws,
    m,
    skip,
):
    """Compute MC/QMC estimators of quantile based measures."""
    qy_alp1 = _unconditional_q_y(
        func,
        n_params,
        loc,
        scale,
        dist_type,
        alp,
        n_draws,
        m,
        skip,
    )
    qy_alp2 = _conditional_q_y(
        func,
        n_params,
        loc,
        scale,
        dist_type,
        alp,
        n_draws,
        m,
        skip,
    )

    # initialization
    q1_alp = np.zeros((len(alp), n_params))
    q2_alp = np.zeros((len(alp), n_params))
    delt = np.array(
        [[np.zeros((1, m)) for x in range(n_params)] for z in range(1)],
        dtype=np.float64,
    )

    # Equation 27 & 28
    for i in range(n_params):
        for pp in range(len(alp)):
            delt[0, i] = qy_alp2[0, i, pp, :] - qy_alp1[pp]  # delt
            q1_alp[pp, i] = np.mean(np.absolute(delt[0, i]))  # |delt|
            q2_alp[pp, i] = np.mean(delt[0, i] ** 2)  # (delt)^2

    return q1_alp, q2_alp


def _nomalized_quantile_based_measures(
    func,
    n_params,
    loc,
    scale,
    dist_type,
    alp,
    n_draws,
    m,
    skip,
):
    """Compute MC/QMC estimators of nomalized quantile based measures."""
    q1_alp, q2_alp = _quantile_based_measures(
        func,
        n_params,
        loc,
        scale,
        dist_type,
        alp,
        n_draws,
        m,
        skip,
    )

    # initialize quantile measures arrays.
    q1 = np.zeros(len(alp))
    q2 = np.zeros(len(alp))
    nomalized_q1_alp = np.zeros((len(alp), n_params))
    nomalized_q2_alp = np.zeros((len(alp), n_params))

    # Equation 13 & 14
    for pp in range(len(alp)):
        q1[pp] = np.sum(q1_alp[pp, :])
        q2[pp] = np.sum(q2_alp[pp, :])
        for i in range(n_params):
            nomalized_q1_alp[pp, i] = q1_alp[pp, i] / q1[pp]
            nomalized_q2_alp[pp, i] = q2_alp[pp, i] / q2[pp]

    return nomalized_q1_alp, nomalized_q2_alp
