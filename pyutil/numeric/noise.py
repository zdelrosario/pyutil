from copy import copy
from numpy import zeros, array, where, mean, sqrt, eye
from math import floor, pow

def ecnoise(fval):
    """Determines the noise of a function from the function values

    Usage
        fnoise, level, inform = ecnoise(fval)
    Arguments
        fval = numpy array of function values; function should be evaluated at
               equi-spaced points based on some user-specified width. For
               example, fval may be

               fval = [f(x-3h), f(x-2h), f(x-h), f(x), f(x+h), f(x+2h), f(x+3h)]

               fval must be of at least length 4,
               though len(fval) = 7 is recommended
    Returns
        fnoise = estimate of the function noise, zero if no noise detected
        level  = array of noise estimates from k-th differences
        inform = status flag, values:
                     1=Noise detected
                     2=Noise not detected; h too small. Try h*100 next.
                     3=Noise not detected; h too large. Try h/100 next.

    @pre len(fval) >= 4
    @pre isinstance(fval, np.ndarray)

    Jorge Mor\'{e} and Stefan Wild. November 2009.
    See Jorge Mor\'{e} and Stefan Wild, ''Estimating computational noise'' (2010)
    Translated to Python (Zachary del Rosario) February 2017
    """
    nf = len(fval)
    level = zeros(nf-1)
    dsgn  = zeros(nf-1)
    fnoise = 0.0
    gamma  = 1.0

    # Compute the range of function values
    fmin = min(fval); fmax = max(fval)
    if (fmax-fmin)/max(abs(fmax),abs(fmin)) > 0.1:
        inform = 3
        return fnoise, level, inform

    # Construct the difference table
    for j in range(nf-2):
        for i in range(nf-1-j):
            fval[i] = fval[i+1] - fval[i]

        # h is too small only when half the function values are equal
        if (j==0 and len(where(fval[:-2]==0)[0])>=nf/2):
            inform = 2
            return fnoise, level, inform

        gamma = 0.5*( (j+1)/float(2*(j+1)-1) )*gamma
        # print("gamma = {}".format(gamma))

        # Compute the estimates for the noise level
        # print("l = {}".format(sqrt( gamma*mean(fval[:-1-j]**2) )))
        level[j] = sqrt( gamma*mean(fval[:-1-j]**2) )

        # Determine differences in sign
        emin = min(fval[:-1-j])
        emax = max(fval[:-1-j])
        if (emin*emax < 0.0):
            dsgn[j] = 1

    # DEBUG
    # print("dsgn = {}".format(dsgn))

    # Determine the noise level
    for k in range(nf-4):
        emin = min(level[k:k+2])
        emax = max(level[k:k+2])

        if (emax<=4*emin and dsgn[k]==1):
            fnoise = level[k]
            inform = 1
            return fnoise, level, inform

    # If noise not detected, then h is too large
    inform = 3
    return fnoise, level, inform

def autonoise(fcn,t0,h0=None):
    """Determines the computational noise of a function
    around a base point. Automates the selection of evaluation points

    Usage
        fnoise, level, inform = autonoise(fcn,t0,h0=None)
    Arguments
        fcn = function to study
        t0  = base point for study
        h0  = evaluation stepsize; optional
    Returns
        fnoise = estimate of the function noise, zero if no noise detected
        level  = array of noise estimates from k-th differences
        inform = status flag, values:
                     1=Noise detected
                     2=Noise not detected; backtracking encountered

    See Jorge Mor\'{e} and Stefan Wild, ''Estimating computational noise'' (2010)
    Implemented in Python (Zachary del Rosario) November 2017
    """
    if h0 == None:
        h0 = 1e-10
    inform = 0
    while not (inform == 1):
        fval = array([fcn(t0+n*h0) for n in range(-3,4)])
        fnoise, level, info_tmp = ecnoise(fval)
        if info_tmp == 2:
            ## Backtracking error
            if inform == 3:
                return fnoise, level, 2
            h0 = h0*100
        if info_tmp == 3:
            ## Backtracking error
            if inform == 2:
                return fnoise, level, 2
            h0 = h0/100
        inform = info_tmp

    return fnoise, level, 1

def stepest(fcn,t0,eps_f):
    """Estimates an appropriate finite difference step-size based
    on computational noise

    Usage
        hs, inform = stepest(fcn,t0,eps_f=None)
    Arguments
        fcn   = function to study
        t0    = base point
        eps_f = measured function computational noise
    Returns
        hs     = recommended FD step size
        inform = status flag, values:
                     1 = hs computed; no alerts
                     2 = hs computed; upper bound violated

    Based on Jorge Mor\'{e} and Stefan Wild, ''Estimating derivatives of noisy
    simulations'' (2012).
    Implemented in Python (Zachary del Rosario) November 2017
    """
    tau1 = 100; tau2 = 0.1
    inform = 1

    ## First try
    hA    = pow(eps_f, 0.25)
    fvalA = [fcn(t0+n*hA) for n in range(-1,2)]
    dhA   = abs(fvalA[0] - 2*fvalA[1] + fvalA[2])
    muA   = dhA / hA**2

    ## Check conditions for hA
    if (abs(fvalA[0]-fvalA[1])<=tau2*max(abs(fvalA[0]),fvalA[1])):
        if not (dhA/eps_f >= tau1):
            inform = 2
        return (pow(8,0.25) * sqrt(eps_f / muA)), inform

    ## Second try
    hB    = pow(eps_f/muA, 0.25)
    fvalB = [fcn(t0+n*hB) for n in range(-1,2)]
    dhB   = abs(fvalB[0] - 2*fvalB[1] + fvalB[2])
    muB   = dhB / hB**2

    ## Check conditions for hB
    if (abs(fvalB[0]-fvalB[1])<=tau2*max(abs(fvalB[0]),fvalB[1])):
        if nor (dhB/eps_f >= tau1):
            inform = 2
        return (pow(8,0.25) * sqrt(eps_f / muB)), inform

    ## Third try
    if (abs(muA-muB)<=0.5*muB):
        return (pow(8,0.25) * sqrt(eps_f / muB)), inform

    raise ValueError("No valid stepsize found...")

def multiest(fcn,x0,eps_f):
    """Estimates appropriate finite difference step-sizes
    for a multivariate function, based on computational noise.
    Really just a wrapper for multiple calls of stepest().

    Usage
        H, inform = multiest(fcn,x0,eps_f)
    Arguments
        fcn   = multivariate function to study; fcn : R^n -> R
        x0    = base point for study; len(x0) == n
        eps_f = measured function computational noise
    Returns
        H      = list of fd stepsizes; len(H) == n
        inform = status flag, values:

    @pre isinstance(x0, np.ndarray)

    Based on Jorge Mor\'{e} and Stefan Wild, ''Estimating derivatives of noisy
    simulations'' (2012).
    Implemented in Python (Zachary del Rosario) November 2017
    """
    ## Run for each input
    inform = 1
    m = x0.shape[0]
    H = zeros(m)
    I = eye(m)
    for ind in range(m):
        fcn_tmp = lambda t: fcn(x0+I[ind]*t)
        H[ind], inf_tmp = stepest(fcn_tmp,0.,eps_f)
        inform = max(inform,inf_tmp)
    return H, inform

### Test functions
if __name__ == "__main__":
    import numpy as np
    import pyutil.numeric as ut

    fcn = lambda x: np.linalg.norm(x)
    n = 10

    # Set (random) state
    # np.random.seed(0)
    # xb = np.random.random(n)
    # np.random.seed(103)
    # p  = np.random.random(n); p = p/np.linalg.norm(p)

    # DEBUG -- set fixed state
    xb = np.ones(n)
    p  = np.ones(n); p = p/np.linalg.norm(p)

    m = 8
    h = 1e-14

    # Evaluate
    fval = np.zeros(m+1)
    mid  = floor((m+2)/2)
    for i in range(m+1):
        s = 2*(i+1-mid)/m
        x = xb + s*h*p
        fval[i] = fcn(x)

    # Trial ecnoise()
    fval_c = copy(fval)
    fnoise, level, inform = ecnoise(fval_c)
    rel_noise = fnoise / fval[int(mid)]

    print("fnoise    = {}".format(fnoise))
    print("rel_noise = {}".format(rel_noise))

    # Trial autonoise()
    x0 = np.ones(n)
    fcn_tmp = lambda t: fcn(x0*(1+t))

    eps_f, lev, inform2 = autonoise(fcn_tmp,0)

    print("eps_f     = {}".format(eps_f))

    ## Trial stepest()
    hs, inform3 = stepest(fcn_tmp,0,eps_f)

    print("hs = {0:}".format(hs))

    ## Trial multiest()
    H, inform4 = multiest(fcn,x0,eps_f)

    print("H = {0:}".format(H))

    G = ut.grad(x0,fcn,h=H)
