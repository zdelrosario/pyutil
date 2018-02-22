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

def array_comp(S,m=None):
    """Computes the complement of an integer list
    """
    # Assume m from max element
    if m == None:
        m = max(S)
    # Compute complement
    I = sorted(set(S))  # Sorted and de-duplicated
    mask = np.ones(m,dtype=bool); mask[I] = 0
    Sc = np.arange(m)[mask]

    return Sc

# Currently bugged out! Scaling is wrong, such that elements can add up to more
# than the total variance
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

def dr_sobol(fcn,X1,X2,Y1=None):
    """Computes the total Sobol indices using the MC method described in
    'Polynomial chaos expansion for sensitivity analysis', Crestaux, T. et. al.

    Usage
        S = dr_sobol(fcn,X1,X2)
    Arguments
        fcn = function handle f(x) for x \in R^m
        X1  = sample set 1, numpy array
        X2  = sample set 2, numpy array
        Y1  = function values for X1
    Returns
        S = list of total Sobol indices, ordered as input variables x

    @pre X1.shape == X2.shape === (N,m)
    """
    N,m = X1.shape
    # Generate the X1 sample realizations if necessary
    if Y1 is None:
        Y1 = np.array([ fcn(x) for x in X1 ])
    # Compute the global statistics
    D_0 = np.mean(Y1)**2
    D   = np.var(Y1)
    # Iterate over all input variables
    S = []
    for i in range(m):
        val = 0
        I = [i]
        Ic= array_comp(I,m)
        for j in range(N):
            # Construct inverleaved sample
            x = np.zeros(m)
            x[I] = X2[j,I]
            x[Ic]= X1[j,Ic]
            # Evaluate
            val += Y1[j]*fcn(x)/N
        # Compute total sobol index
        S.append( 1. - (val-D_0)/D )

    return S

if __name__ == "__main__":
    ### Setup
    import numpy as np
    import pyutil.numeric as ut
    import matplotlib.pyplot as plt
    from scipy.linalg import qr

    np.set_printoptions(precision=3)

    n = int(1e3)
    m = 6
    ### Define QoI
    v = np.ones(m) / np.sqrt(m)
    fcn = lambda x: np.sin(2*np.pi*v.dot(x))
    ### Simulate an AS run
    M = np.concatenate((ut.col(v),np.eye(m)),axis=1)
    W,_ = qr(M,mode='full')
    ### Run dr_sobol() on reparameterized fcn
    fcn_p = lambda xi: fcn(W.dot(xi))
    Xi1 = np.random.random((n,m))*2.-1.
    Xi2 = np.random.random((n,m))*2.-1.

    S = ut.dr_sobol(fcn_p,Xi1,Xi2)

    print("S_total = \n{}".format(np.array(S)))

    ### Run dr_sobol() on the Ishigami function

    N = int(1e3)

    ## Setup
    m = 3
    Xi1 = np.random.random((N,m))*2*np.pi - np.pi
    Xi2 = np.random.random((N,m))*2*np.pi - np.pi

    a = 7.0
    b = 0.1

    fcn = lambda xi: np.sin(xi[0]) + a*np.sin(xi[1])**2 + b*xi[2]**4*np.sin(xi[0])

    ## Exact variance components
    D    = (a**2)/8. + (b*np.pi**4)/5. + (b**2*np.pi**8)/18. + 0.5
    D1   = (b*np.pi**4)/5. + (b**2*np.pi**8)/50. + 0.5
    D2   = a**2/8.
    D3   = 0.
    D12  = 0.
    D13  = (b**2*np.pi**8)/18 - (b**2*np.pi**8)/50.
    D23  = 0.
    D123 = 0.

    S_T1 = (D1 + D12 + D13 + D123) / D
    S_T2 = (D2 + D12 + D23 + D123) / D
    S_T3 = (D3 + D13 + D23 + D123) / D

    ## Approximate total Sobol indices
    S_T_hat = dr_sobol(fcn,Xi1,Xi2)

    ## Report
    print("--------------------------------------------------")
    print("S_T1   = {0:6.4f}, S_T2   = {1:6.4f}, S_T3   = {2:6.4f}".format(S_T1,S_T2,S_T3))
    print("S_T1_h = {0:6.4f}, S_T2_h = {1:6.4f}, S_T3_h = {2:6.4f}".format(*S_T_hat))

    # plt.plot(W[:,0].T.dot(Xi1.T),fcn(Xi1.T),'.')
    # plt.show()
