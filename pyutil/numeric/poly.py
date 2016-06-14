"""A collection of utilities for handling
tensor product polynomials
"""

from numpy import array
from scipy.special import legendre
from scipy.special import hermite

from util import multi_index

# Tensor product polynomial evaluation
def tpolyval(P,X):
    """Evaluates a tensor product polynomial
    Usage
        val = tpolyval(P,X)
    Arguments
        P   = 2-D array of polynomial coefficients;
              each column defines a polynomial
        X   = array of scalar values; each
              element is the evaluation point
              for each polynomial in the tensor
              product
    Returns
        val = evaluated polynomial
    """
    # Convert to numpy arrays
    P = np.array(P)
    X = np.array(X)
    # Check shape agreement
    pass
    # Multiply values
    val = 1.0
    for i in range(P.shape[1]):
        val = val * np.polyval(P[:,i],X[i])
    # Return result
    return val

# Tensor product legendre coefficients
def tlegendre(k):
    """Returns the polynomial coefficients 
    corresponding to the given multi-index
    Usage
        F = tlegendre(k)
    Arguments
        k = multi-index
    Returns
        F = list of polynomial objects
    """
    F = []
    for i in k:
        F.append(legendre(i))
    return F

# Evaluates a legendre tensor-product polynomial
def tfcn(F):
    """Creates a tensorized function from
    a list of function handles
    Usage
        fcn = tfcn(F)
    Arguments
        F   = list of function handles
    returns
        fcn = tensor product function handle
    """
    # Define the function
    def fcn(X):
        val = 1.0
        for i in range(len(F)):
            val = val * F[i](X[i])
        return val
    return fcn

# Defines the k-th legendre tensor-product polynomial
def tleg(k):
    """
    """
    F = tlegendre(k)
    return tfcn(F)

# Test code
if __name__ == "__main__":
    from util import multi_index
    import numpy as np
    from integrate import integrate_rule

    import matplotlib.pyplot as plt

    m = 3   # Dimensions
    n = 3   # Order
    K = multi_index([n]*m)
    
    V = np.zeros((len(K),len(K)))
    
    for i in range(len(K)):
        for j in range(len(K)):
            k1 = K[i]; f1 = tleg(k1)
            k2 = K[j]; f2 = tleg(k2)

            prod = lambda x: f1(x)*f2(x)
            V[i,j] = integrate_rule(prod, 10, m, 'LEGENDRE')
    # Plot results
    plt.figure()
    plt.spy(V)
    plt.show()