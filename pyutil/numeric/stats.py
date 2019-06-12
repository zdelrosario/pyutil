from numpy import cov, mean, array, diag, argsort, zeros, eye, sqrt, dot, var, sqrt
from numpy import errstate, true_divide, isfinite, quantile, atleast_1d
from numpy.random import choice, random
from scipy.linalg import svd, eig, solve
from scipy.stats import f, nct, norm

# from .util import norm_col

def div0( a, b ):
    """Numpy vector division a / b
    ignoring divide by zero errors
    Usage
      q = div0( a, b )
    Arguments
      a = numerator
      b = denominator
    Returns
      q = quotient
    """
    with errstate(divide='ignore', invalid='ignore'):
        c = true_divide( a, b )
        c[ ~ isfinite( c )] = 0  # -inf inf NaN
    return c

def slice(Y,X,H):
    # Check input consistency
    n = len(Y)
    if n != X.shape[0]:
        raise ValueError("len(Y) and X.shape[0] do not match!")
    # Sample statistics
    sig_xx = cov(X, rowvar=False) # Columns are variables
    x_bar  = mean(X, axis=0)
    # Find inverse sqrt of sig_xx
    V,L,_ = svd(sig_xx)
    val = div0(1,sqrt(L)); val[L<1e-16] = 0
    sig_ir = V.dot(diag(val).dot(V.T))
    # Standardize
    Z = array([sig_ir.dot(x-x_bar) for x in X])
    # Slice the response
    Ind = argsort(Y)
    num = n / H
    I_h = [Ind[i*num:(i+1)*num] for i in range(H-1)]
    I_h.append(Ind[(H-1)*num:])
    # Compute proportions
    P_h = array( [float(len(idx)) for idx in I_h] )

    return Z, P_h, I_h, sig_ir, n

def dr_sir(Y,X,H=15):
    """Sliced Inverse Regression
    'Sliced Inverse Regression for Dimension Reduction', K. Li
    Usage
      B,L = dr_sir(Y,X)
    Arguments
      Y = Response samples
      X = Predictor samples, rows are samples
    Keyword Arguments
      H = number of slices
    Returns
      B = Estimate of Central Subspace
      L = Eigenvalues of M_h
    """
    # Slice range
    Z, P_h, I_h, sig_ir, n = slice(Y,X,H)
    # Compute slice sample means
    D_h = array( [mean(Z[I_h[i]],axis=0) for i in range(H)] )
    # Compute weighted PCA
    M_h = D_h.T.dot( diag(P_h).dot( D_h) )
    L,W = eig(M_h)
    # Sort
    Jnd = argsort(L)[::-1]      # Descending
    # Transform
    B_h = norm_col( sig_ir.dot(W[:,Jnd]) )

    return B_h, L[Jnd]

def dr_save(Y,X,H=15):
    """Sliced Average Variance Estimation
    'Marginal tests with sliced average variance estimation',
    Y. Shao, R. Cook, S. Weisberg
    Usage
      B,L = dr_save(Y,X)
    Arguments
      Y = Response samples
      X = Predictor samples, rows are samples
    Keyword Arguments
      H = number of slices
    Returns
      B = Estimate of Central Subspace
      L = Eigenvalues of M_h
    """
    # Slice range
    Z, P_h, I_h, sig_ir, n = slice(Y,X,H)
    # Compute slice sample variances
    S_h = [cov(Z[I_h[i]],rowvar=False) for i in range(H)]
    # Compute weighted PCA
    M_h = zeros(sig_ir.shape)
    I   = eye(sig_ir.shape[0])
    for ind in range(len(S_h)):
        M_h += P_h[ind]/n * (I-S_h[ind]).dot(I-S_h[ind].T)
    L,W = eig(M_h)
    # Sort
    Jnd = argsort(L)[::-1]      # Descending
    # Transform
    B_h = norm_col( sig_ir.dot(W[:,Jnd]) )

    return B_h, L[Jnd]

def hotelling(X):
    # Perform Hotelling's T^2 test on a dataset
    #
    # Usage
    #   pval = hotelling(X)
    # Arguments
    #   X    = columns are vector data
    # Returns
    #   pval = p-value under H0: bar(X) = 0

    xbar = mean(X,axis=1)
    W    = cov(X)
    k, n = X.shape
    t2   = n * dot( xbar, solve(W,xbar) )
    fst  = (n-k)*t2 / ((n-1)*k)
    pval = 1 - f.cdf(fst,k,n-k)

    return pval

def bootstrap_ci(
        X,
        fcn_theta,
        con       = 0.99,
        fcn_se    = None,
        theta_hat = None,
        se        = None,
        n_boot    = 1000,
        n_sub     = 25
):
    """Construct bootstrap confidence intervals
    Implements the "bootstrap-t interval" method discussed in
    Efron and Tibshirani (1993)

    Usage
        theta_l, theta_u = bootstrap_ci(X, fcn_theta, con = con)
    Arguments
        X         = numpy array of (possibly multivariate) samples
        fcn_theta = function which takes a sample X and returns single scalar
                    theta_hat = fcn_theta(X)
        fcn_se    = function which takes sample X and returns se; approximated
                    via additional inner-bootstrap if not provided
        con       = desired confidence level, default con = 0.99
        theta_hat = estimated statistic; will be computed if not given
        se        = standard error for statistic; if not given, estimated from given
                    bootstrap resample
        n_boot    = monte carlo resamples for bootstrap
                    default value n_sub = 1000
        n_sub     = additional resamples used to approximate SE of bootstrap estimate
                    default value n_sub = 25
    Returns
        theta_lo = lower confidence bound
        theta_hi = upper confidence bound

    References and Notes
    - Efron and Tibshirani (1993) "An introduction to the bootstrap"
      "The bootstrap-t procedure... is particularly applicable to location statistics
       like the sample mean.... The bootstrap-t method, at least in its simple form,
       cannot be trusted for more general problems, like setting a confidence interval
       for a correlation coefficient."
    """
    ## Derived quantities
    n_samples = X.shape[0]
    alpha     = (1 - con) / 2

    ## Initial estimate
    if theta_hat is None:
        theta_hat = fcn_theta(X)
    n_entries = len(atleast_1d(theta_hat))

    ## Main loop for bootstrap
    theta_all   = zeros((n_boot, n_entries))
    se_boot_all = zeros((n_boot, n_entries))
    z_all       = zeros((n_boot, n_entries))
    theta_sub   = zeros((n_sub,  n_entries))

    for ind in range(n_boot):
        ## Construct resample
        Ib             = choice(n_samples, size = n_samples, replace = True)
        theta_all[ind] = fcn_theta(X[Ib])

        if fcn_se is None:
            ## Approximate bootstrap se by internal loop
            for jnd in range(n_sub):
                Isub           = Ib[choice(n_samples, size = n_samples, replace = True)]
                theta_sub[jnd] = fcn_theta(X[Isub])

            se_boot_all[ind] = sqrt( var(theta_sub, axis = 0) )
        else:
            se_boot_all[ind] = fcn_se(X[Ib])

        ## Construct approximate pivot
        z_all[ind] = (theta_all[ind] - theta_hat) / se_boot_all[ind]

    ## Compute bootstrap table
    t_lo, t_hi = quantile(z_all, q = [1 - alpha, alpha], axis = 0)

    ## Approximate se of original statistic via bootstrap, if necessary
    if se is None:
        if fcn_se is None:
            se = sqrt( var(theta_all, axis = 0) )
        else:
            se = fcn_se(X)

    ## Construct confidence interval
    theta_lo = theta_hat - t_lo * se
    theta_hi = theta_hat - t_hi * se

    return theta_lo, theta_hi

def rcorr(rk, dim):
    """Build a random correlation matrix
    Usage
        R = rcorr(rk, dim)
    Arguments
        rk  = rank for random vector kernel
        dim = dimension of correlation matrix (dim x dim)
    Returns
        R   = correlation matrix
    """
    W  = random((dim, rk)) * 2 - 1
    S  = dot(W, W.T) + diag(random(dim))
    Sd = diag(1 / sqrt(diag(S)))

    return dot(Sd, dot(S, Sd))

def k_pc(p,c,n):
    # Compute the knockdown factor for a basis value,
    # assuming normally distributed data
    #
    # Usage
    #   k = k_pc(p,c,n)
    # Arguments
    #   p = Desired population fraction (B: 0.90, A: 0.99)
    #   c = Desired confidence level    (B: 0.95, A: 0.95)
    #   n = Number of samples
    # Returns
    #   k = Knockdown factor; B = \hat{X} - k * S

    return nct.ppf(c,n-1,-norm.ppf(1-p)*sqrt(n)) / sqrt(n)

if __name__ == "__main__":
    # Setup
    import numpy as np
    import time
    np.set_printoptions(precision=3)

    # n = int(301)                    # Number of samples
    # H = 15                          # Slices for range
    # # Problem
    # # Test case taken from (Hardle, Simar), Ch. 18.3 SIR
    # m = 3                           # Ambient dimensionality
    # B = np.array([[1,1,1],[1,-1,-1]]).T
    # fcn = lambda x: B[:,0].dot(x) + (B[:,0].dot(x))**3 + 4*(B[:,1].dot(x))**2 + \
    #       np.random.normal()
    # # Data
    # X_s = np.random.normal(size=(n,m))
    # Y_s = np.array([fcn(x) for x in X_s])
    # # Test SIR
    # B_sir, L_sir = dr_sir(Y_s,X_s)
    # # Test SAVE
    # B_save, L_save = dr_save(Y_s,X_s)

    # # Print results
    # print("")
    # print("Results:")
    # print("L_sir  = {}".format(L_sir))
    # print("L_save = {}".format(L_save))
    # print("B_sir  = \n{}".format(B_sir))
    # print("B_save = \n{}".format(B_save))
    # print("")

    ## Test bootstrap table CI method
    # --------------------------------------------------
    np.random.seed(101)
    n_dim = 2
    mu    = np.arange(n_dim)
    sig   = 1

    con    = 0.90
    n_samp = 50
    n_rep  = 100

    fcn_theta  = lambda X: [np.mean(X[:, 0]), np.mean(X[:, 1])]
    theta_true = mu

    theta_lo_all = np.zeros(n_rep)
    theta_hi_all = np.zeros(n_rep)

    X = np.zeros((n_rep, n_samp, n_dim))
    X[:, :, 0] = np.random.normal(size = (n_rep, n_samp), loc = mu[0], scale = sig)
    X[:, :, 1] = np.random.normal(size = (n_rep, n_samp), loc = mu[1], scale = sig)
    theta_hat = fcn_theta(X)

    n_elem = len(np.atleast_1d(theta_hat))

    theta_lo_all = np.zeros((n_rep, n_elem))
    theta_hi_all = np.zeros((n_rep, n_elem))
    bool_cover   = np.zeros((n_rep, n_elem))

    t0 = time.time()
    for ind in range(n_rep):
        theta_lo_all[ind], theta_hi_all[ind] = bootstrap_ci(
            X[ind],
            fcn_theta,
            con    = con
            # fcn_se = fcn_se
        )
        bool_cover[ind] = (theta_lo_all[ind] <= theta_true) * \
                          (theta_true <= theta_hi_all[ind])

    t1 = time.time()

    coverage_obs = np.mean(bool_cover, axis = 0)

    print("Execution time: {0:4.3f}".format(t1 - t0))
    print("Observed coverage = {}".format(coverage_obs))
