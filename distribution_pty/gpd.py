

def genpareto_gradient_cdf(x, c, scale):
    """Gradient of the Generalized Pareto Distribution function w.r.t. to the scale and shape parameter

    :param x: array_like
        quantiles
    :param c:  positive number
        shape parameter
    :param scale:positive number
        scale parameter (default=1)
    :return: (2 X n)-matrix where n is equal to the size of x
        The first row  corresponds to the gradient of the cdf w.r.t. the shape parameter evaluated at x
        The second row corresponds to the gradient of the cdf w.r.t. the scale parameter evaluated at x
    """

    output = np.zeros(shape=(2, x.size))

    cond = 0 < (1+c*x/scale)

    output[0] = np.where(cond, (-1/c**2*log(1 + c*x/scale) + x/(c*(scale + c * x)))*(1 - genpareto.cdf(x, c, scale)), 0)
    output[1] = -x/scale*genpareto.pdf(x, c, scale)

    return output

genpareto.gradientcdf = genpareto_gradient_cdf


def genpareto_fit_ci(h, hGrad, alpha=0.05, verbose=True):
    """
    Build (1-alpha) confidence intervals of functions of GPD parameters

    This function applies the delta method to derive a (1-alpha) confidence interval
    of the function h(c, scale) where c, and scale are respectively the MLE estimators
    of the shape and scale parameters of a GPD distribution.

   :param h: function
            Function taking for arguments the shape c, and scale parameters of a GPD distribution
    :param hGrad: function
            Gradient of the function h w.r.t to the shape c, and scale
    :param alpha: scalar
            level of confidence
    :param verbose:  boolean
            Indicates whether messages should be printed on the screen
    :return: data frame
            contains three values:
             * lB:  lower bound of the confidence interval
             * hHat: estimated value of h(c,scale)
             * uB:  upper bound of the confidence interval
    """
    c = 1 #TBC
    scale = 1  # TBC

    Fnu = 1 #TBC
    nexc = 10 #TBC
    u = 10 #YBC

    Sigma = (1+c)*np.matrix([[1+c, -scale], [-scale, 2*scale**2]])

    # Point estimate of h for the given estimation of c and scale
    hHat = h(c, scale)
    if c < -1:
        output = pd.DataFrame({
                               'lB': [float('nan')],
                               'hHat': [float('nan')],
                               'uB': [float('nan')]
                               })

    if c >= -1:
        if nexc < 30 & verbose:
            print("Unreliable Delta-Method for u =" + str(u) + ". Nexc < 30")

        if c < -1/2 & verbose:
            print("Unreliable Delta-Method for u =" + str(u) + ". c < -1/2")

        gradient = hGrad(c, scale)
        sdHat = sqrt(gradient.T*Sigma*gradient/nexc)
        ci = hHat + norm.ppf([alpha/2, 1-alpha/2])*sdHat

        output = (1-Fnu)*pd.DataFrame({'lB': [ci[0]], 'hHat': [hHat], 'uB': [ci[1]]})

    return output

genpareto.fitci = genpareto_fit_ci