from numpy import array, exp, power, sqrt
from numpy.polynomial import hermite
from scipy.special import factorial

alp = 1.

def lam_gaussian(e, n):
    """
    Returns the desired eigenvalue of the gaussian kernel

    Usage
      lam = lam_gaussian(e, n)
    Arguments
      e   = scaling coefficient in K(x, y) = exp(-e**2 * (x - y)**2)
      n   = index
    Returns
      lam = n-th eigenvalue for parameterized gaussian kernel

    References
      Gregory E. Fasshauer, "Positive Definite Kernels: Past, Present and Future"
    """

    return alp * power(e, 2 * n) / \
        power(0.5 * alp**2 * ( \
            1 + sqrt(1 + power(2 * e / alp, 2)) \
          ) + power(e, 2), \
          n + 0.5 \
        )

def phi_gaussian(e, n, X):
    """
    Evaluates the desired eigenfunction of the gaussian kernel

    Usage
      Phi = phi_gaussian(e, n, X)
    Arguments
      e   = scaling coefficient in K(x, y) = exp(-e**2 * (x - y)**2)
      n   = index
      X   = array of function input points
    Returns
      Phi = array of function values

    References
      Gregory E. Fasshauer, "Positive Definite Kernels: Past, Present and Future"
    """
    c = 1 + power(2 * e / alp, 2)
    k = power(c, 0.125) / sqrt(2**n * factorial(n)) * \
      exp(-(sqrt(c) - 1) * alp**2 * power(X, 2) / 2.)
    Xi = power(c, 0.25) * alp * X

    return k * hermite.hermval(Xi, [0] * (n - 1) + [1])

if __name__ == "__main__":
    ## Simple test
    import numpy as np

    X = np.linspace(-1, +1)
    e = 1 / 10.

    l0 = lam_gaussian(e, 0)
    P0 = phi_gaussian(e, 0, X)

    ## TODO: Verify implementation!
