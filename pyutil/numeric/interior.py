# Standard libraries
import numpy as np
from scipy.optimize import fmin_bfgs

# Custom libraries
from barrier import log_barrier, inv_barrier, ext_obj

def constrained_opt(F,G,x0,tol=1e-8,it_max=1e3,disp=False):
    """ Constrained Optimization via interior point method
    Usage
        xs, Fs, Gs, X, Ft = constrained_opt(F,G,x0)
    Arguments
        F       = objective function, R^n->R
        G       = leq constraints G >= 0, R^n->R^k
        x0      = initial guess
    Keyword Arguments
        tol     = convergence tolerance (L2 gradient)
        it_max  = maximum iterations
    Outputs
        xs      = optimal point
        Fs      = optimal value
        Gs      = gradient at optimal point
        X       = sequence of iterates
        Ft      = sequence of function values
    """
    ### Setup
    r   = 1e2       # Initial relaxation
    r_max = 1e3
    fac = 2         # Relaxation factor
    eps = 1/r       # Initial gradient tolerance
    err = tol*2     # Initial error

    it  = 0     # iteration count
    s   = 1e-1  # interior slack
    x0  = np.array(x0)  # initial guess
    n   = np.size(x0)   # dim of problem

    ### Feasibility problem
    G0     = lambda x: ext_obj(G(x),s)
    # Minimize G0
    xs, _ = fmin_bfgs(G0,x0,retall=True,disp=disp)
    X  = np.array([xs])
    Ft = [F(xs)+log_barrier(G(xs))/r]
    it = it + 1

    ### Interior point problem sequence
    while (err > tol) and (it<it_max):  # Not converged
        # Relax the barrier
        fcn = lambda x: F(x) + log_barrier(G(x))/r
        # Enforce a tighter convergence criterion
        res = fmin_bfgs(fcn,xs,retall=True,gtol=eps,epsilon=1e-8,full_output=True,disp=disp)
        it = it+1
        xn = res[0]; Xn = res[-1]
        Gs = res[2]
        X  = np.append(X,Xn,axis=0)
        Ft = Ft + [fcn(x) for x in Xn]
        # Increment to next problem
        if r < r_max:
            r   = r * fac
            eps = 1 / r
        else:
            r   = r_max
            eps = eps=np.finfo(float).eps
        # Compute error
        err = np.linalg.norm(xn-xs)
        # Set new start guess
        xs = xn

    # Convergence messages
    print("constrained_opt status:")
    if it >= it_max:
        print("    Maximum iteration count reached.")
    if err <= tol:
        print("    Error tolerance met.")
    else:
        print("    Error tolerance not met!")

    Fs = F(xs)
    ### Terminate
    return xs, Fs, Gs, X, Ft
