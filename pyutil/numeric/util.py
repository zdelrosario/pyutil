"""Collection of numeric utility functions
"""

# Necessary imports
from scipy.linalg import svd
from scipy import compress, transpose
from numpy import pad
from numpy.linalg import norm
from numpy import dot
from numpy import atleast_2d
from numpy import ravel
from numpy import reshape
from numpy import zeros, shape, nonzero
from math import log
from numpy import diag
import copy

##################################################
# Computation
##################################################
# Quadratic form
def quad(x,A,y=[]):
    """Computes the quadratic x^T A y
    Usage
        res = quad(x,A)
        res = quad(x,A,y)
    Inputs
        x   = column vector
        A   = square matrix
        y   = column vector, defaults to x
    Returns
        res = x^T A y
    """
    if not any(y):
        return (x.T.dot(A)*x.T).sum(axis=1)[0]
    else:
        return (x.T.dot(A)*y.T).sum(axis=1)[0]

##################################################
# Linear Algebra
##################################################

# Nullspace computation
# from scipy.linalg import svd
# from scipy import compress, transpose
def null(A, eps=1e-15):
    """Computes a basis for the nullspace of a matrix
    Usage
        N = null(A)
    Arguments
        A = rectangular matrix
    Keyword Arguments
        eps = singular value tolerance for nullspace detection
    Returns
        N = matrix of column vectors; basis for nullspace
    """
    u, s, vh = svd(A)
    s = pad(s,(0,vh.shape[0]-len(s)),mode='constant')
    null_mask = (s <= eps)
    null_space = compress(null_mask, vh, axis=0)
    return transpose(null_space)

##################################################
# Reshaping
##################################################

# Return a 2D numpy array with column vectors
# from numpy import atleast_2d
def col(M):
    """Returns column vectors
    Usage
        Mc = col(M)
    Inputs
        M  = 1- or 2-dimensional array-like
    Returns
        Mc = 2-dimensional numpy array
    """
    res = atleast_2d(M)
    # We're not ok with row vectors!
    # Transpose back to a column vector
    if res.shape[0] == 1:
        res = res.T
    return res

# Vectorize a matrix
# from numpy import ravel
def vec(A):
    """Vectorize a matrix
    Usage
        v = vec(A)
    Arguments
        A = matrix of size n x m
    Returns
        v = vector of size n*m, stacked 
            columns of A
    """
    return col(ravel(A))

# Unvectorize a matrix
# from numpy import reshape
def unvec(v,n):
    """Unvectorizes an array
    Usage
        M = unvec(v,n)
    Inputs
        v = vector of length l=n*m
        n = number of rows for M
    Returns
        M = matrix of size n x m
    """
    return reshape(v,(n,-1))

##################################################
# Measurement
##################################################

# Subspace distance
# from numpy.linalg import norm
# from numpy import dot
def subspace_distance(W1,W2):
    """Computes the subspace distance
    Note that provided matrices must 
    have orthonormal columns
    Usage
        res = subspace_distance(W1,W2)
    Inputs
        W1 = orthogonal matrix
        W2 = orthogonal matrix
    Returns
        res = subspace distance
    """
    return norm(dot(W1,W1.T)-dot(W2,W2.T),ord=2)

# Active Subspace Dimension
def as_dim(Lam,eps=2.5):
    """Finds the dimension of the 
    most accurate Active Subspace
    Usage
        dim = as_dim(Lam)
    Arguments
        Lam = positive Eigenvalues, sorted by decreasing magnitude
    Keyword Arguments
        eps = order of magnitude gap needed for AS
    Returns
        dim = dimension of most accurate Active Subspace
    """
    # Normalize by eigenvalue energy
    s = sum(Lam)
    L = [l/s for l in Lam]
    # Check eigenvalue gaps
    Gp= [log(L[i]/L[i+1],10) for i in range(len(L)-1)]
    # If no gap exceeds eps, full dimensional
    gap = max(Gp)
    if gap < eps:
        return len(L)
    # Return dimension based on largest gap
    else:
        return Gp.index(gap)+1

##################################################
# Normalization
##################################################

# Normalize the columns of a matrix
# from numpy import diag
def norm_col(M):
    """Normalizes the columns of a matrix
    Usage
        Mn = norm_col(M)
    Arguments
        M  = 2-dimensional numpy array (matrix)
    Returns
        Mn = 2-dimensional numpy array with same shape
             as M, whose columns have an L2 norm of 1
    """
    E = zeros(M.shape)
    for i in range(M.shape[1]):
        n = norm(M[:,i])
        if n > 0:
            E[:,i] = M[:,i] / n
        else:
            pass
    return E

    # E = diag( [1/norm(M[:,i]) for i in range(M.shape[1])] )
    # return M.dot(E)

# Rounds by smallest non-zero magnitude vector element
# from numpy import zeros, shape, nonzero
def round_out(M):
    """Attempts to round a matrix to recover integer values
    Usage
        Mr = round_out(M)
    Arguments
        M  = 2-dimensional numpy array
    Returns
        Mr = a copy of M where each column has been
             divided by the smallest nonzero element
    """
    C = zeros(shape(M))
    for i in range(shape(M)[1]):
        c = min(min(M[nonzero(M[:,i]),i]))
        C[:,i] = M[:,i] / c
    return C

##################################################
# Dimensionality
##################################################

# Generate the full list of multi-indices
# import copy
def multi_index(N):
    """Generates list of elements in a multi-index
    Usage
        K = multi_index(N)
    Arguments
        N = list of elements per dimension;
            length is number of dimensions
    Returns
        K = list of multi-index values
    """
    d = len(N)
    # Seed with first element
    res = [[0] * d]
    # Loop over every dimension
    for i in range(d):
        # Create buffer for added rows
        add_on = []
        # Loop over every existing row, skip first
        for j in range(len(res)):
            # Loop over every possible value
            for k in range(N[i]):
                temp = copy.copy(res[j]); temp[i] = k
                add_on.append(temp)
        # Replace with buffer
        res = add_on
    return res

##################################################
# Test code
##################################################
if __name__ == "__main__":
    import numpy as np

    ### Test vec and unvec
    # A = np.arange(9); A.shape=(3,3)
    # v = vec(A)
    # M = unvec(v,3)
    # print(M)

    ### Test as_dim
    # Lam1 = [1,0.9] # Should be 2D
    # Lam2 = [1e3,1e1,0.5e1,1e0] # should be 1D
    # Lam3 = [1e3,0.9e3,1e1,1e0] # should be 2D
    # Lam4 = [1e3,0.9e3,0.8e3,1e0] # should be 3D
    # Lam5 = [1e3,0.9e3,0.8e3,0.7e3] # should be 4D
    
    # print("AS 1 dim={}".format(as_dim(Lam1)))
    # print("AS 2 dim={}".format(as_dim(Lam2)))
    # print("AS 3 dim={}".format(as_dim(Lam3)))
    # print("AS 4 dim={}".format(as_dim(Lam4)))
    # print("AS 5 dim={}".format(as_dim(Lam5)))

    ### Test subspace_distance
    # M = np.

    ### Test column normalization
    # M = np.reshape(np.arange(9),(3,3))
    # Mn= norm_col(M)
    # print( Mn )

    ### Test quad
    x = col([1]*3)
    A = np.arange(9).reshape((-1,3))
    y = col([2]*3)

    print("x^TAx = {}".format(quad(x,A)))
    print("x^TAy = {}".format(quad(x,A,y)))
