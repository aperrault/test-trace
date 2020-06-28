import numpy as np
import load_contact_data
import scipy.integrate

fit_params = [8.32667268e-16, 1.55750000e+01]
counts_of_duration = np.array([np.sum(load_contact_data.contact_durations == i) for i in range(1, 6)])

# convert stretched exponential exp(-a * x ** b) to censored probabilities
def get_censored_probs(a, b):
    # convert minutes to hours
    intervals = [(0, 5 / 60.), (5 / 60., 15 / 60.), (15 / 60., 1.), (1., 4.), (4., np.inf)]
    # integrate
    raw_probs = np.array([scipy.integrate.quad(lambda x: np.exp(-a * x ** b), interval[0], interval[1])[0] for interval in intervals])
    return raw_probs / np.sum(raw_probs)


# params = (a, b)
def neg_log_likelihood(params):
    censored_probs = get_censored_probs(params[0], params[1])
    log_likelihood = 0.
    for i in range(5):
        log_likelihood += counts_of_duration[i] * np.log(np.exp(-params[0] * censored_probs[i] ** params[1]))
    return -log_likelihood

def optimize():
    return = scipy.optimize.minimize(neg_log_likelihood, [0.8, 8], method='Nelder-Mead')
