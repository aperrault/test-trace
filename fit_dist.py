import math
import scipy.optimize
import numpy as np
import scipy.stats
import code

def nbinom_pdf(val, n_successes, p_success):
    return scipy.stats.nbinom.pmf(val, n_successes, p_success)

def poisson_pdf(val, param):
    return scipy.stats.poisson.pmf(val, param)

def neg_log_likelihood_nbinom(params, values):
    nll = 0.

    cutoff = max(values) + 1
    vals = np.array(range(cutoff))

    nbinom_lik = nbinom_pdf(vals, params[0], params[1])

    for i in range(np.shape(values)[0]):
        nll += -np.log(nbinom_lik[values[i]])

    return nll

def neg_log_likelihood_poisson(params, values):
    nll = 0.

    cutoff = max(values) + 1
    vals = np.array(range(cutoff))

    poisson_lik = poisson_pdf(vals, params[0])

    for i in range(np.shape(values)[0]):
        nll += -np.log(poisson_lik[values[i]])

    return nll

def fit_lognorm_moments(mean, std, median):
    wrapper = (lambda params: (scipy.stats.lognorm.mean(s=params[0], loc=params[1], scale=params[2]) - mean) ** 2 + (scipy.stats.lognorm.std(s=params[0], loc=params[1], scale=params[2]) - std) ** 2 + (scipy.stats.lognorm.median(s=params[0], loc=params[1], scale=params[2]) - median) ** 2)
    res = scipy.optimize.minimize(wrapper, [1., 1., 1.], method='Nelder-Mead')
    return res.x

def fit_neg_binom(data):
    wrapper = (lambda params: neg_log_likelihood_nbinom(params, data))
    res = scipy.optimize.minimize(wrapper, [1.518016, 0.892663], method='Nelder-Mead')
    mean_dispersion = ((1 - res.x[1]) * res.x[0]/res.x[1], res.x[0])
    return res.x, mean_dispersion

def fit_poisson(data):
    wrapper = (lambda params: neg_log_likelihood_poisson(params, data))
    res = scipy.optimize.minimize(wrapper, [1.518016], method='Nelder-Mead')
    return res.x
