"""Gaussian process fit
"""

##################################################
# Import block
##################################################

import numpy as np
from .util import col, quad, vec, unvec
from numpy.random import rand, randn
from scipy.linalg import inv
from numpy import sqrt


def fit_gp(Y,X,T=[],P=[],full=False):
    """Fits a gaussian process to data
    Usage
        y_hat = fit_gp(Y,X)
        y_hat, s_hat = fit_gp(Y,X,full=True)
    Arguments
        Y = function values, array of length n
        X = query points, array of shape (m,n)
    Keyword Arguments
        T = weight parameters, array of length m, T_i >= 0
        P = power parameters, array of length m, P_i \in [1,2]
        full = full output flag
    Outputs
        y_hat = best linear unbiased estimator, function handle
        s_hat = mean-squared error, function handle
    """

    # Dimension of problem
    m = X.shape[0]
    n = X.shape[1]

    # Check agreement
    if len(Y) != n:
        raise ValueError("Y and X of incompatible sizes!")

    # Assumed correlation parameters
    if not any(T):
        T = col([1.0] * m)
    if not any(P):
        P = col([2] * m)

    # Helper one vector
    O = col([1] * n)

    # Helper functions
    def dist(x,y):
        """Weighted distance
        """
        d = 0
        for i in range(m):
            d = d + T[i]*abs(x[i]-y[i])**P[i]
        return d

    def corr(x,y):
        """Correlation function
        """
        return np.exp(-dist(x,y))

    def compute_r(X):
        """Compute covariance matrix
        """
        R = np.zeros((n,n))
        # Iterate over lower triangle
        for i in range(n):
            for j in range(i,n):
                R[i,j] = corr(X[:,i],X[:,j])
                R[j,i] = R[i,j]
        return R

    def mu_hat(y,Ri):
        """Mean estimator
        """
        return quad(O,Ri,y) / quad(O,Ri,O)

    def sig2_hat(y,Ri,mu,n):
        """Variance estimator
        """
        v = y-O*mu
        return quad(v,Ri)/n

    def y_fcn(y,mu,X,Ri,xs):
        """Best linear unbiased estimator
        """
        r = col([corr(xs,X[:,i]) for i in range(n)])
        return mu + quad(r,Ri,col(y-O*mu))

    def s2(sig2,X,Ri,xs):
        """Mean-squared error
        """
        r = col([corr(xs,X[:,i]) for i in range(n)])
        return sig2*( 1 - quad(r,Ri,r) + (1-quad(O,Ri,r))**2/quad(O,Ri,O) )

    # Compute fit
    R = compute_r(X)
    Ri= inv(R)

    mu   = mu_hat(Y,Ri)
    sig2 = sig2_hat(Y,Ri,mu,n)

    # Best linear unbiased estimator
    y_hat = lambda x: y_fcn(Y,mu,X,Ri,x)
    if full:
        s_hat = lambda x: s2(sig2,X,Ri,x)
        return y_hat, s_hat
    else:
        return y_hat
