from copy import copy
from numpy import zeros, array, where, mean, sqrt
from math import floor

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

    Jorge More' and Stefan Wild. November 2009.
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

### Test ecnoise()
if __name__ == "__main__":
    import numpy as np

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

    # Call the noise detector
    fval_c = copy(fval)
    fnoise, level, inform = ecnoise(fval_c)
    rel_noise = fnoise / fval[int(mid)]

    print("fnoise    = {}".format(fnoise))
    print("rel_noise = {}".format(rel_noise))
