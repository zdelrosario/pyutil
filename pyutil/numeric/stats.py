from numpy import cov, mean, array, diag, argsort, zeros, eye
from numpy import errstate, true_divide, isfinite
from scipy.linalg import svd, eig

def div0( a, b ):
    """Numpy vector division a / b 
    ignoring divide by zero errors
    Usage
      q = div0( a, b )
    Arguments
      a = numerator
      b = denominator
    Returns
      q = quotient
    """
    with errstate(divide='ignore', invalid='ignore'):
        c = true_divide( a, b )
        c[ ~ isfinite( c )] = 0  # -inf inf NaN
    return c

def slice(Y,X,H):
    # Check input consistency
    n = len(Y)
    if n != X.shape[0]:
        raise ValueError("len(Y) and X.shape[0] do not match!")
    # Sample statistics
    sig_xx = cov(X, rowvar=False) # Columns are variables
    x_bar  = mean(X, axis=0)
    # Find inverse sqrt of sig_xx
    V,L,_ = svd(sig_xx)
    val = div0(1,L); val[L<1e-16] = 0
    sig_ir = V.dot(diag(val).dot(V.T))
    # Standardize
    Z = array([sig_ir.dot(x-x_bar) for x in X])
    # Slice the response
    Ind = argsort(Y)
    num = n / H
    I_h = [Ind[i*num:(i+1)*num] for i in range(H-1)]
    I_h.append(Ind[(H-1)*num:]) 
    # Compute proportions
    P_h = array( [float(len(idx)) for idx in I_h] )

    return Z, P_h, I_h, sig_ir

def dr_sir(Y,X,H=15):
    """Sliced Inverse Regression
    'Sliced Inverse Regression for Dimension Reduction', K. Li
    Usage
      B,L = dr_sir(Y,X)
    Arguments
      Y = Response samples
      X = Predictor samples, rows are samples
    Keyword Arguments
      H = number of slices
    Returns
      B = Estimate of Central Subspace
      L = Eigenvalues of M_h
    """
    # Slice range
    Z, P_h, I_h, sig_ir = slice(Y,X,H)
    # Compute slice sample means
    D_h = array( [mean(Z[I_h[i]],axis=0) for i in range(H)] )
    # Compute weighted PCA
    M_h = D_h.T.dot( diag(P_h).dot( D_h) )
    L,W = eig(M_h)
    # Sort
    Jnd = argsort(L)[::-1]      # Descending
    # Transform
    B_h = sig_ir.dot(W[:,Jnd])

    return B_h, L[Jnd]

def dr_save(Y,X,H=15):
    """Sliced Average Variance Estimation
    'Marginal tests with sliced average variance estimation',
    Y. Shao, R. Cook, S. Weisberg
    Usage
      B,L = dr_save(Y,X)
    Arguments
      Y = Response samples
      X = Predictor samples, rows are samples
    Keyword Arguments
      H = number of slices
    Returns
      B = Estimate of Central Subspace
      L = Eigenvalues of M_h
    """
    # Slice range
    Z, P_h, I_h, sig_ir = slice(Y,X,H)
    # Compute slice sample variances
    S_h = [cov(Z[I_h[i]],rowvar=False) for i in range(H)]
    # Compute weighted PCA
    M_h = zeros(sig_ir.shape)
    I   = eye(sig_ir.shape[0])
    for ind in range(len(S_h)):
        M_h += P_h[ind]/n * (I-S_h[ind]).dot(I-S_h[ind].T)
    L,W = eig(M_h)
    # Sort
    Jnd = argsort(L)[::-1]      # Descending
    # Transform
    B_h = sig_ir.dot(W[:,Jnd])

    return B_h, L[Jnd]

if __name__ == "__main__":
    # Setup
    import numpy as np
    n = int(301)                    # Number of samples
    H = 15                          # Slices for range
    # Problem
    # Test case taken from (Hardle, Simar), Ch. 18.3 SIR
    m = 3                           # Ambient dimensionality
    B = np.array([[1,-1,-1],[1,1,1]]).T
    fcn = lambda x: B[:,0].dot(x) + (B[:,0].dot(x))**3 + 4*(B[:,1].dot(x))**2 + \
          np.random.normal()
    # Data
    X_s = np.random.normal(size=(n,m))
    Y_s = np.array([fcn(x) for x in X_s])
    # Test SIR
    B_sir, L_sir = dr_sir(Y_s,X_s)
    # Test SAVE
    B_save, L_save = dr_save(Y_s,X_s)
