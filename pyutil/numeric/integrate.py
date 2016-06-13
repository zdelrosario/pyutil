import numpy as np
import copy
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

# Generate the full list of multi-indices
def index_square(n,d):
    # Seed with first element
    res = [[0] * d]
    # Loop over every dimension
    for i in range(d):
        # Create buffer for added rows
        add_on = []
        # Loop over every existing row, skip first
        for j in range(len(res)):
            # Loop over every possible value
            for k in range(n):
                temp = copy.copy(res[j]); temp[i] = k
                add_on.append(temp)
        # Replace with buffer
        res = add_on
    return res

# Partial product of weights
def part_prod(fcn,Xi,x,w):
    scale = 1
    X_c = []
    for i in range(len(Xi)):
        scale = scale * w[Xi[i]]
        X_c.append(x[Xi[i]])
    return scale * fcn(X_c)

# Generic m-D cubature function
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
    # Load 1-D points/weights
    x, w = rule(n)
    # Generate multi-index list
    E = index_square(n, m)
    # Iterative summation
    res = part_prod(fcn,E[0],x,w)
    for i in range(1,len(E)):
        res += part_prod(fcn,E[i],x,w)
    return res

def integrate_hermite(fcn, n, m):
    return cubature(fcn, n, m, hermgauss)

def integrate_legendre(fcn, n, m):
    return cubature(fcn, n, m, leggauss)

def integrate_rule(fcn, n, m, flag):
    """Integrates an m-dimensional function on [-1,1]
    via a tensor-product cubature rule
    Usage
        res = integrate_rule(f, n, m, flag)
    Arguments
        fcn = function to integrate
        n   = number of points to use per dimension
        m   = dimension of parameter space
        flag= quadrature rule to employ: HERMITE or LEGENDRE
    Returns
        res = result of cubature
    """
    if flag.upper() == 'HERMITE':
        return integrate_hermite(fcn, n, m)
    elif flag.upper() == 'LEGENDRE':
        return integrate_legendre(fcn, n, m)
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
