from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes

def genSeed(i,length=256):
    """Generates an integer for a random seed, applying a
    cryptographic hash function to an integer input,
    following the recommendation of A.B. Owen, "Monte
    Carlo theory, methods and examples"

    Usage
      j = genSeed(i,length=256)
    Arguments
      i = integer value
      length = bytelength of converted integer
    Returns
      j = integer value

    @post j \in [0,2^32-1]
    """

    b = i.to_bytes(length, byteorder='big', signed=False)
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(b)
    s = digest.finalize()

    return int.from_bytes(s, byteorder='big', signed=False) % (2**32-1)

if __name__ == "__main__":
    ### Test MC tools
    import numpy as np
    from scipy.stats import t
    import time
    np.set_printoptions(precision=2)

    ### Test the seed generator
    # Generate random strings from consecutive generated seeds,
    # check for significance in their correlations
    N = int(1e4) # Length of random strings
    M = 3        # Number of strings
    B = int(1e3) # Bootstrap resamples
    alp = 0.01

    Seeds = [0] * M
    X_all = np.zeros((N,M))

    # Generate from sequential seeds
    # for ind in range(M):
    #     Seeds[ind] = genSeed(ind)

    #     np.random.seed(Seeds[ind]);
    #     X_all[:,ind] = np.random.random(N)

    # Generate from single seed
    X_all = np.random.random((N,M))

    # Check the rng stream correlations
    def corrMat(data):
        c = np.cov(data,rowvar=False)
        n = np.diag(1/np.sqrt(np.diag(c)))

        return np.dot(n,np.dot(c,n))
    R = corrMat(X_all)

    # Perform bootstrap hypothesis test
    R_boot = np.zeros((M,M,B))
    M_boot = np.zeros((M,M,B))

    t0 = time.time()
    for ind in range(B):
        I = np.random.choice(N,N)
        R_boot[:,:,ind] = corrMat(X_all[I,:])
        M_boot[:,:,ind] = R_boot[:,:,ind] - R
    R_boot.sort(axis=-1)
    M_mat = np.mean( np.abs(M_boot), axis=-1 )
    S_mat = np.sqrt( np.var(R_boot, axis=-1) ) / np.sqrt(B)
    T_mat = M_mat / S_mat
    p_mat = 1 - t.cdf( T_mat, B-1 )
    t1 = time.time()

    # Compute empirical CI
    R_lo = R_boot[:,:,int(B * alp/2)]
    R_hi = R_boot[:,:,int(B * (1-alp/2))]

    print("--------------------------------------------------")
    print("Execution time: {0:6.4f}".format(t1-t0))
    print("R     = \n{}".format(R))
    print("R_lo  = \n{}".format(R_lo))
    print("R_hi  = \n{}".format(R_hi))
    print("p_mat = \n{}".format(p_mat))
