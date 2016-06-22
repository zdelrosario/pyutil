# Standard libraries
# from math import log, exp
import numpy as np
from numpy.linalg import norm
from ad import adnumber
from ad.admath import log, exp

def inv_barrier(g):
    """Inverse barrier function
    Takes a constraint evaluation, and builds a scalar inverse barrier
    Usage
        res = inv_barrier(g)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
    Outputs
        res = value of inverse barrier for given values
    """
    res = 0
    for val in g:
        res = res - 1/val
    return res


def log_barrier(g):
    """Log barrier function
    Takes a constraint evaluation, and builds a scalar log barrier
    Usage
        res = inv_barrier(g)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
    Outputs
        res = value of log barrier for given values
    """
    res = 0
    for val in g:
        if val < -1:
            pass
        else:
            if val >= 0:
                res = float('inf')
            else:
                res = res - log(-val)
    return res

def ext_obj(g,e):
    """Exterior Point Objective
    Creates an objective function for the exterior point method
    Usage
        res = ext_obj(g,e)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
        e = scalar value, represents interior distance from boundary sought
    Outputs
        res = value of objective function
    """
    res = 0
    for val in g:
        res = res + max(0,val+e)**2
    return res

def feasible(g):
    """Checks if a constraint value is feasible
    Usage
        res = feasible(g)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
    Outputs
        res = boolean feasibility, True/False
    """
    slack = 1e-1
    res = True
    for val in g:
        res = res and (val<0)
    return res

if __name__ == "__main__":
    # # Test ad on an external objective
    # from ad import gh
    # from example1 import g
    # s  = 1e-1
    # G0 = lambda x: ext_obj(g(adnumber(x)),s)
    # dG0, _ = gh(G0)
    # # Return ordinary values
    # G = lambda x: G0(x).x
    # dG= lambda x: dG0(x).T

    # # Evaluate
    # x0 = np.array([2.5,2.5])
    # val = G0(x0)
    # dif = dG0(x0)
    # # Run BFGS
    # from scipy.optimize import minimize
    # res = minimize(G, x0, method='BFGS', jac=dG0)

    # Test ad on interior objective
    from ad import gh
    from example1 import g,f
    s = 1e-1
    fcn = lambda x: f(x) + log_barrier(g(x))/5.0
    dfcn, _ = gh(fcn)
    # Run BFGS
    from scipy.optimize import minimize
    x0 = [0,0]
    res = minimize(fcn, x0, method='BFGS', jac=dfcn)