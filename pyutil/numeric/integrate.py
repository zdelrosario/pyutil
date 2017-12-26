"""Collection of integration utilities intended for
multi-dimensional numerical integration (cubature)
Note that these are for convenience -- they are
not optimized for speed.
"""

import numpy as np
import copy
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from operator import mul

from util import multi_index

# Convenience function
def prod(L):
    """Returns the product of an iterable's elements
    Usage
        r = prod(L)
    Arguments
        L = an iterable of elements with binary multiplication
    Returns
        r = result of product
    """
    return reduce(mul, L, 1)

# Partial product of weights
def part_prod(fcn,Xi,x,w):
    scale = 1
    X_c = []
    for i in range(len(Xi)):
        scale = scale * w[Xi[i]]
        X_c.append(x[Xi[i]])
    return scale * fcn(X_c)

# Currently takes fixed number of points per dimension,
# could extend to vary points per dimension easily
def cubature_data(n,m,rule):
    """
    Usage
        X, W = cubature_data(n,m,rule)
    Arguments
        n    = points per dimension
        m    = dimensions
        rule = quadrature nodes of form {x, w = rule(n)}
    Returns
        X    = quadrature nodes
        W    = quadrature weights
    """
    # Load 1-D nodes and weights
    x, w = rule(n)
    # Generate multi-index list
    E = multi_index([n]*m)
    # Generate m-D nodes and weights
    X = []; W = []
    for e in E:
        X.append([x[i] for i in e])
        W.append(prod([w[i] for i in e]))
    # Return
    return X, W

# Currently takes fixed number of points per dimension,
# could extend to vary points per dimension easily
def cubature(fcn, n, m, rule):
    """Perform cubature based on a provided 
    quadrature rule
    Usage
        res = cubature(fcn, n, m, rule)
    Arguments
        fcn = function to integrate
        n   = number of points to use per dimension
        m   = dimension of parameter space
        rule= function handle which returns quadrature nodes &
              weights, given the desired number of nodes
    Returns
        res = result of cubature
    """
    # Load 1-D nodes and weights
    x, w = rule(n)
    # Generate multi-index list
    E = multi_index([n]*m)
    # Iterative summation
    res = part_prod(fcn,E[0],x,w)
    for i in range(1,len(E)):
        res += part_prod(fcn,E[i],x,w)
    return res

def cubature_hermite(fcn, n, m):
    return cubature(fcn, n, m, hermgauss)

def cubature_legendre(fcn, n, m):
    return cubature(fcn, n, m, leggauss)

def cubature_rule(fcn, n, m, flag):
    """Integrates an m-dimensional function on [-1,1]
    via a tensor-product cubature rule
    Usage
        res = cubature_rule(f, n, m, flag)
    Arguments
        fcn = function to integrate
        n   = number of points to use per dimension
        m   = dimension of parameter space
        flag= quadrature rule to employ: HERMITE or LEGENDRE
    Returns
        res = result of cubature
    """
    if flag.upper() == 'HERMITE':
        return cubature_hermite(fcn, n, m)
    elif flag.upper() == 'LEGENDRE':
        return cubature_legendre(fcn, n, m)
    else:
        raise ValueError("Quadrature rule '"+flag+"' unrecognized")

# Scales and precomposes for QR on [-1,1]^m
def normalize_function(fcn, L, U):
    """Scales and shifts a function for gauss cubature
    Usage
        fp  = normalize_function(f, L, U)
    Arguments
        f   = function handle
        L   = Lower bound iterable
        U   = Upper bound iterable
    Returns
        fp  = scaled and shifted version of f for
              cubature on [-1,1]^m
    """
    if len(L) != len(U):
        raise ValueError("Dimensions of L and U do not match!")
    m = len(L)
    A = []; b = []; C = 1
    for i in range(m):
        A.append( (U[i]-L[i])/2. )
        b.append( (U[i]+L[i])/2. )
        C = C * A[i]
    A = np.diag(np.array(A))
    b = np.array(b)
    return lambda x: C*fcn(A.dot(np.array(x))+b)
