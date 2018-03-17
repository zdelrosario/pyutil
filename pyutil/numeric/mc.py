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
    np.set_printoptions(precision=2)

    ### Test the seed generator
    N = int(1e6)
    M = 10

    Seeds = [0] * M
    X_all = np.zeros((M,N))
    S_all = np.zeros(M)

    for ind in range(M):
        Seeds[ind] = genSeed(ind)

        np.random.seed(Seeds[ind]); 
        X_all[ind] = np.random.random(N)
        S_all[ind] = np.sqrt(np.var(X_all[ind]))

    # Check the rng stream correlations
    C_all = np.zeros((M,M))
    for ind in range(M):
        for jnd in range(ind):
            C_all[ind,jnd] = np.cov(X_all[ind],X_all[jnd])[0,1]
            C_all[jnd,ind] = C_all[ind,jnd]

    Sinv  = np.diag(1 / S_all)
    R_all = np.dot(Sinv,np.dot(C_all,Sinv))
    np.fill_diagonal(R_all,1)

    print("R_all = {}".format(R_all))
