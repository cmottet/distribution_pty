import numpy as np
from scipy.stats import norm
from numpy import
from numpy import (log, exp)

#
# !!!! Lognormal distribution parametrization python is not the smae as in R !!!!
# Have to update this... Arf...
#
#

def lognorm_dcdf(x, d, s=1, loc=0):
    """ d^th derivative of the cumulative distribution function at x of the given RV.

    :param x:  array_like
        quantiles
    :param d: positive integer
        derivative order of the cumulative distribution function
    :param s: positive number
        s parameter (default=1)
    :return: array_like
     If d = 0: the cumulative distribution function evaluated at x
     If d = 1: the probability density function evaluated at x
     If d => 2: the (d-1)-density derivative evaluated at x

     :Examples:
     lognorm.dcdf(1,1,0,2)
    """
    if d < 0 | (not isinstance(d, int)):
        print("d must be a non-negative integer.")
        return float('nan')

    if d == 0:
        output = lognorm.cdf(x, loc=loc, s=s)

    if d == 1:
        output = lognorm.pdf(x, loc=loc, s=s)

    if d == 2:
        output = np.where(x > 0, -1/(s*x)**2*norm.pdf((log(x)-loc)/s)*((log(x) - loc) / s + s), 0)

    if d == 3:
        def deriv3(x, s):
            trans_x = (log(x) - loc) / s + s
            return 1 / (x * s)**3*norm.pdf((log(x)-loc)/s) * (trans_x**2 + s * trans_x - 1)

        output = np.where(x > 0, deriv3(x, s), 0)

    if d > 3:
        print("Not available for this package version")
        output = np.where(x > 0, np.repeat(float('nan'), x.__len__), 0)

    return output


def partial_expectation(x, s=1, loc=0, lower=True):
    """
    Compute E[X I(X <= x)] when the r.v. X has a log-normal distribution

    :param x: array_like
        quantiles
    :param loc:
    :param s: positive number
        s parameter (default=1)
    :param lower: Boolean
        If true, returns E[X I(X <= x)] and E[X I(X > x)] otherwise
    :return:
    :Example:
    partial_expectation(1, 0,1)
    """
    mean = exp(loc + s**2/2)
    output = np.where(lower,
                      mean*norm.cdf((log(x) - loc)/s - s),
                      mean*(1 - norm.cdf((log(x) - loc)/s - s))
                      )
    return output


lognorm.dcdf = lognorm_dcdf
lognorm.partial_expectation = partial_expectation
