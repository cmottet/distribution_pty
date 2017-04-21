import numpy as np
from scipy.stats import (lognorm, norm)
from numpy import (log, exp)


def lognorm_dcdf(x, d, loc=0, scale=1):
    """ d^th derivative of the cumulative distribution function at x of the given RV.

    :param x:  array_like
        quantiles
    :param d: positive integer
        derivative order of the cumulative distribution function
    :param scale: positive number
        scale parameter (default=1)
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
        output = lognorm.cdf(x, loc=loc, s=scale)

    if d == 1:
        output = lognorm.pdf(x, loc=loc, s=scale)

    if d == 2:
        output = np.where(x > 0, -1/(scale*x)**2*norm.pdf((log(x)-loc)/scale)*((log(x) - loc) / scale + scale), 0)

    if d == 3:
        def deriv3(x, scale):
            trans_x = (log(x) - loc) / scale + scale
            return 1 / (x * scale)**3*norm.pdf((log(x)-loc)/scale) * (trans_x**2 + scale * trans_x - 1)

        output = np.where(x > 0, deriv3(x, scale), 0)

    if d > 3:
        print("Not available for this package version")
        output = np.where(x > 0, np.repeat(float('nan'), x.__len__), 0)

    return output


def partial_expectation(x, loc=0, scale=1, lower=True):
    """
    Compute E[X I(X <= x)] when the r.v. X has a log-normal distribution

    :param x: array_like
        quantiles
    :param loc:
    :param scale: positive number
        scale parameter (default=1)
    :param lower: Boolean
        If true, returns E[X I(X <= x)] and E[X I(X > x)] otherwise
    :return:
    :Example:
    partial_expectation(1, 0,1)
    """
    mean = exp(loc + scale**2/2)
    output = np.where(lower,
                      mean*norm.cdf((log(x) - loc)/scale - scale),
                      mean*(1 - norm.cdf((log(x) - loc)/scale - scale))
                      )
    return output


lognorm.dcdf = lognorm_dcdf
lognorm.partial_expectation = partial_expectation
