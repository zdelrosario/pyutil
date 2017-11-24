from math import floor
import numpy as np
from scipy.stats import norm

def halton(n,b):
    # Generates a Halton sequence of requested length and basis
    # Usage
    #    r = halton(n,b)
    # Arguments
    #    n = samples requested, must be integer > 0
    #    b = basis for sequence, should be prime*
    # Returns
    #    r = list of points in [0,1]
    #
    # Notes
    #    *If multiple sequences are to be used to fill a d-dimensional space,
    #    the bases used should be coprime.
    #
    #    If operating in high dimensions, note that subsequent primes may give
    #    highly correlated sequences. This can be addressed by scrambling the
    #    sequence, or using the leaped Halton sequence (skipping points).

    return [halton_kern(i,b) for i in range(1,n+1)]

def halton_kern(i,b):
    f = 1.
    r = 0.

    # Generate sequence
    while i > 0:
        f = f / b
        r = r + f * (i % b)
        i = floor( float(i)/b )

    return r

def rwh_primes1(n):
    """ Returns a list of primes < n
    By Robert William Hanks
    via http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188

    Usage:
        P = rwh_primes1(n)
    Arguments:
        n = maximum value
    Returns
        P = list of primes < n"""
    sieve = [True] * (n/2)
    for i in xrange(3,int(n**0.5)+1,2):
        if sieve[i/2]:
            sieve[i*i/2::i] = [False] * ((n-i*i-1)/(2*i)+1)
    return [2] + [2*i+1 for i in xrange(1,n/2) if sieve[i]]


def rwh_primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n
    By Robert William Hanks
    via http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/3035188#3035188

    Usage:
        P = rwh_primesfrom2to(n)
    Arguments:
        n = maximum value
    Returns
        P = list of primes < n"""
    sieve = np.ones(n/3 + (n%6==2), dtype=np.bool)
    for i in xrange(1,int(n**0.5)/3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k/3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)/3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]

def qmc_unif(n,m,seed=0):
    """Generate a low-discrepancy (Halton) sequence on [0,1]^m

    Usage:
        Xi = qmc_unif(n,m)
    Arguments:
        n  = number of points
        m  = dimension of domain
    Keyword Arguments:
        seed = permutation random seed
    Returns:
        Xi = numpy array of m-dimensional points in [0,1]^m
    """
    # Prime basis
    if m <= 26:
        P = rwh_primesfrom2to(101)
    # TODO -- handle higher-dimensional cases

    Xi = np.zeros((n,m))

    for i in range(m):
        Xi[:,i] = halton(n,P[i])
    # TODO -- scramble Halton sequence for larger primes

    return Xi

def qmc_norm(n,m,mu=None,var=None):
    """Generate a pseudo-normal sequence from a Halton sequence

    Usage:
        Xi = qmc_normal(n,m)
    Arguments:
        n  = number of points
        m  = dimension of domain
    Keyword Arguments:
        mu  = mean of distribution, list of length m
        var = variance of distribution, list of length m*
    Returns:
        Xi = numpy array of m-dimensional points in [-Inf,Inf]^m

    Notes:
        *We assume a diagonal variance structure
    """
    # Handle default arguments
    if mu is None:
        mu = np.zeros(m)
    if var is None:
        var = np.ones(m)
    # Generate a uniform Halton sequence in [0,1]^m
    Xi = qmc_unif(n,m)
    # Map via inverse cdf
    for i in range(m):
        Xi[:,i] = norm.ppf(Xi[:,i], loc=mu[i], scale=np.sqrt(var[i]))
    return Xi

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = int(1e3)
    Xi_u = 2*qmc_unif(n,2) - 1
    Xi_n = qmc_norm(n,2,var=[1.0,0.5])

    plt.figure()
    plt.plot(Xi_u[:,0],Xi_u[:,1],'bo')
    plt.plot(Xi_n[:,0],Xi_n[:,1],'ro')
    plt.xlim((-4,4)); plt.ylim((-4,4))

    plt.show()
