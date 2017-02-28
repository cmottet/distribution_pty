def lognorm_dpdf(x, d, loc=0, scale=1):
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
        print("d must be a non-negative integer.")
        return float('nan')

    if d == 0:
        output = lognorm.cdf(x, loc=loc, s=scale)

    if d == 1:
        output = lognorm.pdf(x, loc=loc, s=scale)

    if d == 2:
        output = np.where(x > 0, -1/(scale*x)**2*lognorm.pdf(log(x), loc=loc, s=scale)*((log(x) - loc) / scale + scale), 0)

    if d == 3:
        def deriv3(x, scale):
            trans_x = (log(x) - loc) / scale + scale
            return 1 / (x * scale) ** 3 * lognorm.pdf(log(x), loc=loc, s=scale) * (trans_x**2 + scale * trans_x - 1)

        output = np.where(x > 0, deriv3(x, scale), 0)

    if d > 3:
        print("Not available for this package version")
        output = np.where(x > 0,numpy.repeat(float('nan'), x.__len__),0)

    return output