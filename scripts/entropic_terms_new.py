import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import binom

# -----------------------
# Binary entropy (numeric)
# -----------------------
def binh(p):
    """
    Function computes entropy of a binary random variable given probability of success, P, using the binary entropy function.

    Parameters:
        p: Probability of success
    Returns:
        h: Binary entropy expressed in bits
    """
    p = np.clip(p, 1e-12, 1-1e-12) # to avoid P = 0 or P = 1
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

# -----------------------
# KL divergence term
# -----------------------
def divterm(v,p):
    """
    Function computes the Kullback-Leiber divergence penalty term used in the finite-size security bound for entanglement-based BB84.
    This term corresponds to the divergence D(q || \tilde{ν}_C) appearing in Eqn (145) of arXiv:2405.05912.

    Returns:
        KL divergence penalty term in the finite-size security bound expressed in bits.

    """
    v1, v2, v3 = v
    p1, p2, p3 = p

    return (
        cp.kl_div(v1, p1)
      + cp.kl_div(v2, p2)
      + cp.kl_div(v3, p3)
    ) / np.log(2)

def find_delta(n, p, alpha=0.01):
    # Find quantile
    k = binom.ppf(alpha, n, p)
    
    delta = p - k/n
    return delta

# -----------------------
# Convex optimization: multi-point linear underestimator
# -----------------------

def hterm(n, aldelt, gamma, qberthresh, n_p):
    """
    Function computes the worst-case entropy-related term appearing in the finite-sized keyrate bound for EB-BB84 by carrying out convex optimisation to return the optimal value of the convex security bound from Eqn 145, along with the optimiser-chosen parameters that realise the worst-case eavesdropping scenario.

    Parameters:
        alphahat: Finite-sized security parameter controlling the strength of the statistical penalty. Smaller values lead to more conservative but lower keyrate bounds, and vice versa.
        gamma: Fraction of signals used for testing the channel rather than generating the final key.
        qberthresh: Maximum error rate that the protocol is willing to tolerate before the generated key is considered insecure.
        n_p: Number of tangent points used to approximate nonlinear entropy-like functions.
    Returns:
        prob_value: The optimal value of the convex optimisation problem.
        v1.value, v2.value, perr.value: The optimiser-selected values of the variables that achieved the worst-base bound.
    """

    bdelt = aldelt / (2*aldelt + 1) #bdelt = beta -1
    delta = find_delta(n, 1 - gamma)

    convergence_criteria = 1e-9 #Tolerance for the solution of the convex optimisation to converge

    # CVXPY variables
    v1   = cp.Variable(nonneg=True)
    v2   = cp.Variable(nonneg=True)
    perr = cp.Variable()
    t    = cp.Variable()  # epigraph for the product term

    # ------------------------
    # Tangent Points
    # ------------------------
    p_grid     = np.linspace(1e-12, 1 - 1e-12, n_p)   # perr

    # ------------------------
    # Base constraints
    # ------------------------
    constraints = [
        v1 >= 1e-12,
        v2 >= 1e-12,
        v2 <= qberthresh,
        perr >= 1e-12,
        perr <= 1 - 1e-12,
        1 - gamma*v1 - gamma*v2 >= 1e-12
    ]

    # ------------------------
    # Tangent Lines
    # ------------------------
    for pi in p_grid:
        # Value at reference point
        h = (1-pi)**(1-bdelt) + pi**(1-bdelt)
        f_pi = (1 - gamma - delta) * (1 - (1 / bdelt) * np.log2(h))

        # Gradient at that point
        df_dp = (1 - gamma - delta) * (-1/(bdelt*np.log(2))) * (-(1-bdelt)*((1-pi)**(-bdelt)) + (1-bdelt)*(pi**(-bdelt)))/h

        # Epigraph constraint
        t_constraint = f_pi + (perr-pi)*df_dp
        constraints.append(t >= t_constraint)

    # ------------------------
    # KL divergence term
    # ------------------------
    term2 = ((aldelt + 1)/aldelt) * divterm((gamma*v1, gamma*v2, 1 - gamma*v1 - gamma*v2), 
                                      (gamma*(1 - perr), gamma*perr, 1 - gamma))

    # ------------------------
    # Objective
    # ------------------------
    objective = cp.Minimize(t + term2)

    # ------------------------
    # Solve
    # ------------------------
    prob = cp.Problem(objective, constraints)
    prob.solve(
        solver=cp.MOSEK,
        verbose=False,
        mosek_params={
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": convergence_criteria,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": convergence_criteria,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": convergence_criteria
        }
    )

    return prob.value, (v1.value, v2.value, perr.value), prob.status