import math
import scipy.optimize
import numpy as np
import scipy.stats
import code

def nbinom_pdf(val, n_successes, p_success):
    return scipy.stats.nbinom.pmf(val, n_successes, p_success)

def neg_log_likelihood(params, values):
    nll = 0.

    cutoff = max(values) + 1
    vals = np.array(range(cutoff))

    nbinom_lik = nbinom_pdf(vals, params[0], params[1])

    for i in range(np.shape(values)[0]):
        nll += -np.log(nbinom_lik[values[i]])

    return nll

def fit_neg_binom_pdf(data):
    wrapper = (lambda params: neg_log_likelihood(params, data))
    res = scipy.optimize.minimize(wrapper, [1.518016, 0.892663], method='Nelder-Mead')
    return res.x
