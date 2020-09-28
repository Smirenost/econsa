"""Capabilities for computation of Shapley effects.
This module contains functions to estimate shapley effects for models with
dependent inputs.

"""
# import neccesary modules
import itertools
import numpy as np
import pandas as pd
import chaospy as cp
from tqdm import tqdm
from econsa.sampling import cond_mvn
from econsa.sampling import _r_condmvn

def get_shapley(
        method, model, Xall, Xcond, n_perms, n_inputs, n_output, n_outer, n_inner,
):
    """

    This function calculates Shapley effects and their standard errors for
    models with both dependent and independent inputs. We allow for two ways
    to calculate Shapley effects: by examining all permutations of the given
    inputs or alternatively, by randomly sampling permutations of inputs.

    The function is a translation of the exact and random permutation funtions
    in the ``sensitivity`` package in R, and takes the method (exact or random)
    as an argument and therefore estimates shapley effects in both ways.


    Parameters
    ----------
    method : string
           Specifies which method you want to use to estimate shapley effects,
           the ``exact`` or ``random`` permutations method. When the number of
           inputs is small, it is better to use the ``exact`` method, and
           ``random`` otherwise.

    model : string
          The model/function you will calculate the shapley effects on.

    Xall : string (n)
         A function that takes `n´ as an argument and generates a n-sample of
         a d-dimensional input vector.

    Xset : string (n, Sj, Sjc, xjc)
         A function that takes `n, Sj, Sjc, xjc´ as arguments and generates
         a n- sample an input vector corresponding to the indices in `Sj´
         conditional on the input values `xjc´ with the index set `Sjc´.

    n_perms : scalar
            This is an input for the number of permutations you want the model
            to make. For the ``exact`` method, this argument is none as the
            number of permutations is determined by how many inputs you have,
            and for the ``random`` method, this is determined exogeniously.

    n_inputs : scalar
             The number of input vectors for which shapley estimates are being
             estimated.

    n_output : scalar
             Monte Carlo (MC) sample size to estimate the output variance of
             the model output `Y´.

    n_outer : scalar
            The outer Monte Carlo sample size to estimate the cost function for
            `c(J) = E[Var[Y|X]]´.

    n_inner : scalar
            The inner Monte Carlo sample size to estimate the cost function for
            `c(J) = Var[Y|X]´.

    Returns
    -------
    effects : DataFrame
            n dimensional DataFrame with the estimated shapley effects, the
            standard errors and the confidence intervals for the input vectors.

    Sources
    -------
    .. Shapley exact permutations: https://rdrr.io/cran/sensitivity/src/R/shapleyPermEx.R
    .. Shapley random permutations: https://rdrr.io/cran/sensitivity/src/R/shapleyPermRand.R

    Contributor
    -----------
    Linda Maokomatanda

    """
    if (method == 'exact'):
        permutations = list(itertools.permutations(range(n_inputs), n_inputs))
        permutations = [list(i) for i in permutations]

        n_perms = len(permutations)
    else:
        permutations = np.zeros((n_perms,n_inputs), dtype = np.int64)
        for i in range(n_perms):
            permutations[i] = np.random.permutation(n_inputs)

        n_perms = np.int(permutations.shape[0])

    # initiate empty input array for sampling
    model_inputs = np.zeros((n_output + n_perms * (n_inputs - 1) * n_outer * n_inner, n_inputs))
    model_inputs[:n_output, :] = Xall(n_output).T

    for p in tqdm(range(n_perms)):

        perms = permutations[p]
        perms_sorted = np.argsort(perms)

        for j in range(1,n_inputs):
            # set of the 0st-(j-1)th elements in perms
            Sj = perms[:j]
            # set of the jth-n_perms elements in perms
            Sjc = perms[j:]

            # sampled values of the inputs in Sjc
            xjc_sampled = np.matrix(Xcond(n_outer, Sjc, None, None)).T

            for l in range(n_outer):
                xjc = xjc_sampled[l, :]

                # sample values of inputs in Sj conditional on xjc
                sample_inputs = np.matrix(Xcond(n_inner, Sj, Sjc, xjc.flat)).T
                concatenated_sample = np.concatenate((sample_inputs, np.ones((n_inner, 1)) * xjc), axis = 1)
                inner_indices = n_output + p * (n_inputs - 1) * n_outer * n_inner + (j - 1) * n_outer * n_inner + l * n_inner
                model_inputs[inner_indices:(inner_indices + n_inner), :] = concatenated_sample[:, perms_sorted]

    # calculate model output
    output = model(model_inputs)

    # Initialize Shapley, main and total Sobol effects for all players
    shapley_effects = np.zeros(n_inputs)
    shapley_effects_squared = np.zeros(n_inputs)

    # estimate the variance of the model output
    model_output = output[:n_output]
    output = output[n_output:]
    output_variance = np.var(model_output)

    # estimate shapley, main and total sobol effects
    conditional_variance = np.zeros(n_outer)


    for p in tqdm(range(n_perms)):

        perms = permutations[p]
        previous_cost = 0

        for j in range(n_inputs):
            if (j == (n_inputs - 1)):
                estimated_cost = output_variance
                delta = estimated_cost - previous_cost

            else:
                for l in range(n_outer):
                    model_output = output[:n_inner]
                    output = output[n_inner:]
                    conditional_variance[l] = np.var(model_output, ddof = 1)
                estimated_cost = np.mean(conditional_variance)
                delta = estimated_cost - previous_cost

            shapley_effects[perms[j]] = shapley_effects[perms[j]] + delta
            shapley_effects_squared[perms[j]] = shapley_effects_squared[perms[j]] + delta**2

            previous_cost = estimated_cost

    shapley_effects = shapley_effects / n_perms / output_variance
    shapley_effects_squared = shapley_effects_squared / n_perms /(output_variance ** 2)
    standard_errors = np.sqrt((shapley_effects_squared - shapley_effects**2) / n_perms)


    # confidence intervals
    CI_min = shapley_effects - 1.96 * standard_errors
    CI_max = shapley_effects + 1.96 * standard_errors

    col = ['X' + str(i) for i in np.arange(n_inputs) + 1]

    effects = pd.DataFrame(
        np.array([shapley_effects, standard_errors, CI_min, CI_max]),
        index = ['Shapley effects', 'std. errors', 'CI_min', 'CI_max'],
        columns = col,
    ).T

    return effects