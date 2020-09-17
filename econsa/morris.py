"""Calculate morris indices for models with dependent parameters.

We convert frequently between iid uniform, iid standard normal and multivariate
normal variables. To not get confused, we use the following naming conventions:

-u refers to to uniform variables
-z refers to standard normal variables
-x refers to multivariate normal variables.

"""
from multiprocessing import Pool

import chaospy as cp
import numba as nb
import numpy as np
import pandas as pd
from scipy.stats import norm


def elementary_effects(func, params, cov, n_draws, sampling_scheme="sobol", n_cores=1):
    """Calculate Morris Indices of a model described by func.

    The distribution of the parameters is assumed to be multivariate normal, with
    mean ``params["value"]`` and covariance matrix ``cov``.

    The algorithm is based on Ge and Menendez, 2017, (GM17): Extending Morris method for
    qualitative global sensitivity analysis of models with dependent inputs.

    Parameters
    ----------
    func : function
        Function that maps parameters into a quantity of interest.
    params : pd.DataFrame
        DataFrame with arbitrary index. There must be a column
        called value that contains the mean of the parameter distribution.
    cov : pd.DataFrame
        Both the index and the columns are the same as the index
        of params. The covariance matrix of the parameter distribution.
    n_draws : int
        Number of draws
    sampling_scheme : str
        One of ["sobol", "random"]. Default: "sobol"

    Returns
    -------
    mu_ind : float
        Absolute mean of independent part of elementary effects
    sigma_ind : float
        Standard deviation of independent part of elementary effects

    """
    u_a, u_b = _get_uniform_base_draws(n_draws, len(params), sampling_scheme)
    z_a = _uniform_to_standard_normal(u_a)
    z_b = _uniform_to_standard_normal(u_b)
    mean_np = params["value"].to_numpy()
    cov_np = cov.to_numpy()

    dep_samples_ind_x, a_sample_ind_x = _dependent_draws(
        z_a,
        z_b,
        mean_np,
        cov_np,
        "ind",
    )

    dep_samples_corr_x, _ = _dependent_draws(z_a, z_b, mean_np, cov_np, "corr")

    evals_ind = _evaluate_model(func, params, dep_samples_ind_x, n_cores)

    evals_base_ind = _evaluate_model(func, params, a_sample_ind_x, n_cores)

    evals_corr = _evaluate_model(func, params, dep_samples_corr_x, n_cores)

    evals_base_corr = _shift_sample(evals_base_ind, -1)

    deltas = u_b - u_a

    mu_ind, sigma_ind = _calculate_indices(
        evals_ind,
        evals_base_ind,
        deltas,
        params.index,
    )
    mu_corr, sigma_corr = _calculate_indices(
        evals_corr,
        evals_base_corr,
        deltas,
        params.index,
    )

    mu_ind_cum, sigma_ind_cum = _calculate_cumulative_indices(
        evals_ind,
        evals_base_ind,
        deltas,
        params.index,
    )
    mu_corr_cum, sigma_corr_cum = _calculate_cumulative_indices(
        evals_corr,
        evals_base_corr,
        deltas,
        params.index,
    )

    res = {
        "mu_ind": mu_ind,
        "mu_corr": mu_corr,
        "sigma_ind": sigma_ind,
        "sigma_corr": sigma_corr,
        "mu_ind_cum": mu_ind_cum,
        "mu_corr_cum": mu_corr_cum,
        "sigma_ind_cum": sigma_ind_cum,
        "sigma_corr_cum": sigma_corr_cum,
    }

    return res


def _get_uniform_base_draws(n_draws, n_params, sampling_scheme):
    """Get uniform random draws.

    Questions:
    - Should we replace random by quasi-random?
    - Should we de-correlate the result as a finite sample correction.

    Args:
        n_draws (int)
        n_params (int)
        sampling_scheme (str): one of ["sobol", "random"]

    Returns:
        u_a, u_b (np.ndarray): Arrays of shape (n_draws, n_params) with i.i.d draws
            from a uniform [0, 1] distribution.

    """
    if sampling_scheme == "sobol":
        u = cp.generate_samples(order=n_draws * 2 * n_params, rule="S").reshape(
            n_draws,
            -1,
        )
    elif sampling_scheme == "random":
        u = np.random.uniform(low=1e-5, high=1 - 1e-5, size=(n_draws, 2 * n_params))
    else:
        raise ValueError
    u_a = u[:, :n_params]
    u_b = u[:, n_params:]
    # u_a = np.random.uniform(size=(n_draws, n_params))
    # u_b = np.random.uniform(size=(n_draws, n_params))
    return u_a, u_b


def _uniform_to_standard_normal(uniform):
    """Convert i.i.d uniform draws to i.i.d standard normal draws.

    Args:
        uniform (np.ndarray): Can have any shape.

    Returns
        standard_normal (np.ndarray): Same shape as uniform.

    """
    return norm.ppf(uniform)


def _dependent_draws(z_a, z_b, mean, cov, kind):
    """Create dependent draws for EE^ind with radial design.

    This calculates the two terms of the numerator of equation 39 in GM17, i.e.
    conditions on all but one element.

    The generation of the dependent random samples is simplified because we condition
    on all but one elements. Therefore, and due to the normality assumption, converting
    the re-drawn element to the desired distribution is just rescaling with the standard
    deviation of the conditional distribution and adding the conditional mean.

    Args:
        z_a (np.ndarray): Array of shape (n_draws, n_params) with the "a-sample"
            converted to i.i.d standard normal distribution.
        z_b (np.ndarray): Array of shape (n_draws, n_params) with the "b-sample"
            converted to a i.i.d standard normal distribution.
        mean (np.ndarray): Array of length n_params
        cov (np.ndarray): Array of shape (n_params, n_params)
        kind (str): one of ["ind", "corr"]

    Returns:
        ab_sample_x (np.ndarray): Array of shape (n_draws, n_params, n_params) with the
            left term from the numerator of equation 39 in GM17.
        a_sample_x (np.ndarray): Array of shape (n_draws, n_params, n_params) with the
            right term from the numerator of equation 39 in GM17

    """
    # extract dimensions
    n_draws, n_params = z_a.shape

    if kind == "ind":
        shift = np.arange(n_params).astype(int) + 1
        diag_pos = -1
    elif kind == "corr":
        shift = np.arange(n_params).astype(int)
        diag_pos = 0
    else:
        raise ValueError

    unshift = n_params - shift

    # generate the shifted a sample
    a_sample_z = np.repeat(z_a, n_params, axis=0).reshape(n_draws, n_params, n_params)

    a_sample_z_shifted = _shift_sample(a_sample_z, shift.reshape(1, -1))

    ab_sample_z_shifted = a_sample_z_shifted.copy()
    ab_sample_z_shifted[:, :, diag_pos] = z_b

    shifted_covs = np.zeros((n_params, n_params, n_params))
    for p, s in enumerate(shift):
        shifted_covs[p] = _shift_cov(cov, s)

    # calculated shifted means (one per parameter). They are aligned with shifted covs.
    means = np.repeat(mean.reshape(1, n_params), n_params, axis=0)
    shifted_means = _shift_sample(means, np.arange(n_params, dtype=int) + 1)

    # convert the shifted a sample to a multivariate normal distribution
    shifted_chols = np.linalg.cholesky(shifted_covs)
    a_sample_x_shifted = (
        np.matmul(
            shifted_chols,
            a_sample_z_shifted.reshape(n_draws, n_params, n_params, 1),
        ).reshape(n_draws, n_params, n_params)
        + shifted_means
    )

    ab_sample_x_shifted = (
        np.matmul(
            shifted_chols,
            ab_sample_z_shifted.reshape(n_draws, n_params, n_params, 1),
        ).reshape(n_draws, n_params, n_params)
        + shifted_means
    )
    # un-shift the shifted samples.
    a_sample_x = _shift_sample(a_sample_x_shifted, unshift.reshape(1, -1))
    ab_sample_x = _shift_sample(ab_sample_x_shifted, unshift.reshape(1, -1))

    return ab_sample_x, a_sample_x


@nb.guvectorize(
    ["f8[:], i8, f8[:]", "i8[:], i8, i8[:]"],
    "(m), () -> (m)",
    nopython=True,
)
def _shift_sample(sample, k, out):
    """Re-sort sample such that the first k elements are moved to the end.

    If sample is multidimensional this works on the last axis.

    This corresponds to the function tau_1 in section 3.3 of GM17 but can also
    be used as tau_3 if called with a different k.

    guvectorize is not used for speed, but to get automatic broadcasting.

    Args:
        sample (np.ndarray): Array of shape [..., m]
        k (int): 0 <= k <= m

    Returns:
        shifted (np.ndarray): Same shape as sample.

    """
    m = len(sample)
    for old_pos in range(k):
        new_pos = m - k + old_pos
        out[new_pos] = sample[old_pos]

    for new_pos, old_pos in enumerate(range(k, m)):
        out[new_pos] = sample[old_pos]


def _shift_cov(cov, k):
    """Re-sort a covariance matrix such that the fist k elements are moved to the end.

    Args:
        cov (np.ndarray): Two dimensional array of shape (m, m)
        k (int): 0 <= k <= m

    Returns:
        shifted (np.ndarray): Same shape as cov.

    """
    m = len(cov)
    old_order = np.arange(m).astype(int)
    new_order = _shift_sample(old_order, k).astype(int)
    return cov[new_order][:, new_order]


def _evaluate_model(func, params, sample, n_cores):
    """Do all model evaluations needed for the EE indices.

    Args:
        func (function): Maps params to quantity of interest.
        params (pd.DataFrame): Model parameters. The "value" column will be replaced
            with values from the morris samples.
        sample (np.ndarray): Array of shape (n_draws, n_params, n_params).
            Morris samples in the multivariate normal space.

    Returns:
        evals (np.ndarray): Array of shape (n_draws, n_params) with model evaluations

    """
    n_draws, n_params, _ = sample.shape
    evals = np.zeros((n_draws, n_params))

    inputs = []
    for d in range(n_draws):
        for p in range(n_params):
            par = params.copy()
            par["value"] = sample[d, p]
            inputs.append(par)

    p = Pool(processes=n_cores)
    evals_flat = p.map(func, inputs)

    # evals_flat = Parallel(n_jobs=n_cores)(delayed(func)(inp) for inp in inputs)

    evals = np.array(evals_flat).reshape(n_draws, n_params)

    return evals


def _calculate_indices(evals, evals_a, deltas, params_index):
    """Calculate the morris index.

    Args:
        evals (np.ndarray): Array of shape (n_draws, n_params). This is
            equal to evals_ind or evals_full.
        evals_a (np.ndarray): Array of shape (n_draws, nparams)
        deltas (np.ndarray): Array of shape (n_draws, n_params)
        params_index (pd.Index or pd.MultiIndex): Index of params.

    Returns:
        mu (pd.Series): Absolute mean of elementary effect of each parameter
        sigma (pd.Series): Absolute SD of elementary effect of each parameter.
    """
    ee = np.abs((evals - evals_a) / deltas)
    mu = pd.Series(data=ee.mean(axis=0), index=params_index)
    sigma = pd.Series(data=ee.std(axis=0), index=params_index)
    return mu, sigma


def _calculate_cumulative_indices(evals, evals_a, deltas, params_index):
    """

    Returns:
        mu (pd.DataFrame): Cumulative absolute mean of elementary effect of each parameter.
            there is one column per parameter.
        sigma (pd.DataFrame): ...

    """
    n_draws, _ = evals.shape
    ee = np.abs((evals - evals_a) / deltas)

    ee_df = pd.DataFrame(data=ee, columns=params_index, index=np.arange(n_draws) + 1)
    cum_mu = ee_df.rolling(n_draws, min_periods=2).mean()
    cum_sigma = ee_df.rolling(n_draws, min_periods=2).std()

    return cum_mu, cum_sigma
