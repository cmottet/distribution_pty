from scipy.stats import pareto
import numpy as np



def pareto_dcdf(x, d, b=1, scale=1):
    """ d^th derivative of the cumulative distribution function at x of the given RV.

    :param x:  array_like
        quantiles
    :param d: positive integer
        derivative order of the cumulative distribution function
    :param b: positive number
        shape parameter (default=1)
    :param scale: positive number
        scale parameter (default=1)
    :return: array_like
     If d = 0: the cumulative distribution function evaluated at x
     If d = 1: the probability density function evaluated at x
     If d => 2: the (d-1)-density derivative evaluated at x
    """
    if scale <= 0 or b <= 0:
        print("The scale and shape parameters must be positive numbers.")

    if d == 0:
        output = pareto.cdf(x, b, scale=scale)

    if d != 0:
        output = np.where(scale <= x, -(-1/scale)**d*(x/scale)**(-b-d)*np.prod(b+range(d)), 0)

    return output


def pareto_truncmom(a, m, b=1, scale=1):
    """
    Compute the expected value of the distribution for values larger than a

    :param a: scalar
    :param m: non-negative integer
        order of the moment
    :param b: positive number
        shape parameter (default=1)
    :param scale: positive number
        scale parameter (default=1)
    :return:
    """
    return np.where(m < b, b*scale**b/(b-m)*max(a, scale)**(m-b), float("inf"))

pareto.dcdf = pareto_dcdf
pareto.truncmom = pareto_truncmom