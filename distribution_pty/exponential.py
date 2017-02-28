from scipy.stats import (expon, lognorm, genpareto, norm)
import pandas as pd
import numpy as np
from numpy import (log, sqrt, prod)

def expon_dpdf(x, d, scale=1):
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
    """

    if d < 0 | (not isinstance(d, int)):
            print("D must be a non-negative integer.")
            return float('nan')

    if d == 0:
            output = expon.cdf(x, scale=scale)

    if d >= 1:
            output = ((-1/scale) ** (d - 1)) * expon.pdf(x, scale=scale)

    return output

expon.dpdf = expon_dpdf