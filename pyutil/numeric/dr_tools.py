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
      dist = smoothness values

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
        # Add normalized distance
        dist.append(r / ran)

    return dist
