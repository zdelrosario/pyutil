import numpy as np
from scipy.spatial import cKDTree

def dr_smoothness(Y,X,k=6):
    """Measures quality of a dimension reduction
    via the smoothness of the reparameterization
    Usage
      res = dr_smoothness(Y,X)
    Arguments
      Y = response variable
      X = re-parameterized predictor variables
    Keyword Arguments
      k = number of neighbors for computation,
          default k = 6, bound 1 < k
    Returns
      dist = smoothness value

    @pre X.shape == (n,m)
    @pre len(Y) == n
    """
    # Get variability of Y
    ran = max(Y) - min(Y)
    # Build and query KDTree
    n = X.shape[0]; m = X.shape[1]
    tree = cKDTree(X, leafsize=200)
    dist = []
    # Compute nn-distances
    for ii in range(n):
        N = tree.query(X[ii], k=k)
        r = sum( [abs(Y[ii]-Y[jj]) for jj in N[1]] ) / k
        A = sum( [np.linalg.norm(X[ii]-X[jj]) for jj in N[1]] ) / k
        # Add normalized distance
        dist.append(r) # F variability
        # dist.append(r / ran) # F variability normalized by range
        # dist.append(r / A) # F variability normalized by average distance
        # dist.append(r / A / ran)

    return sum( [d/n for d in dist] )

def inter(*arg):
    """Perform an intersection on an
    arbitrary number of lists
    Usage
        res = inter(I_1,...,I_N)
    Arguments
        I_i = a list
    Returns
        res = the list intersection of all I_i
    """
    # Ensure large enough
    if len(arg) > 1:
        res = np.intersect1d(arg[0],arg[1])
        # Base case
        if len(arg) == 2:
            return res
        # Recurse
        else:
            return inter(res,*arg[2:])
    else:
        raise ValueError("Must provide at least two lists")

def dr_anova(Y,X,k=5):
    """Measures quality of a dimension reduction
    by accounting for variance contributions
    in the dr coordinates
    Usage
        res = dr_anova(Y,X)
    Arguments
        Y = response variable
        X = re-parameterized predictor variables
    Keyword Arguments
        k = number of bins per dimension
    Returns
        res = list of variance fractions, ordered as X

    @pre X.shape == (n,m)
    @pre len(Y) == n
    """
    n,m = X.shape
    # np.array() for vector index access
    Y = np.array(Y)
    # Order samples in each coordinate
    I = []
    for i in range(m):
        I.append( np.argsort(X[:,i]) )
    # Perform index slice
    num = int(n/k)
    Is= []
    for i in range(m):
        Is.append( [I[i][j*num:(j+1)*num] for j in range(k-1) ] )
        Is[i].append( I[i][(k-1)*num:] )
    # Means
    Y_g = np.mean(Y) # Grand mean
    Y_m = [ [np.mean(Y[ind]) for ind in Ij] for Ij in Is ]
    N_m = [ [len(ind) for ind in Ij] for Ij in Is ]
    # Total sum of squares
    # S_t = np.var(Y) * (n-1)
    S_t = np.sum([ (y-Y_g)**2 for y in Y ])
    # Component sum of squares
    S = []
    for i in range(m):
        # S.append( k**(m-1)*N_m[i][j]/S_t * np.sum([ (Y_m[i][j]-Y_g)**2 for j in range(k) ]) )
        S.append( np.sum([ (Y_m[i][j]-Y_g)**2 for j in range(k) ]) )

    return S
