import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
import cvxpy as cp
from scipy.stats import binom

from general_optimisation import trialanderror

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
# Applying the first Renyi Entropy Formula
# -----------------------

# -----------------------
# Convex optimization: multi-point linear underestimator
# -----------------------

def hterm1(n, aldelt, gamma, qberthresh, n_p):
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

    alpha = aldelt + 1 #convert back to alpha
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
    p_grid     = np.linspace(1e-12, 2/3 - 1e-12, n_p)   # perr

    # ------------------------
    # Base constraints
    # ------------------------
    constraints = [
        v1 >= 1e-12,
        v2 >= 1e-12,
        v2 <= qberthresh,
        perr >= 1e-12,
        perr <= 2/3 - 1e-12,
        1 - gamma*v1 - gamma*v2 >= 1e-12
    ]

    # ------------------------
    # Tangent Lines
    # ------------------------
    for pi in p_grid:
        # Value at reference point
        h = (
            (2**(1 - (1/alpha))) * (pi)
            + (1 - pi)**(1 - (1/alpha))
            * ((pi/2)**(1/alpha)
            + (1 - (3*pi/2))**(1/alpha)
            )
            )

        f_pi = (1 - gamma - delta) * ((alpha / (1 - alpha)) * np.log2(h) + 1)

        # Gradient at that point
        h_diff = (
        (2**(1 - (1/alpha)))

        - (1 - (1/alpha))
        * (1 - pi)**(-1/alpha)
        * (
        (pi/2)**(1/alpha)
        + (1 - (3*pi/2))**(1/alpha)
        )

        + (1 - pi)**(1 - (1/alpha))
        * (
        (1/(2*alpha))
        * (pi/2)**((1/alpha) - 1)

        - (3/(2*alpha))
        * (1 - (3*pi/2))**((1/alpha) - 1)
        )
        )

        df_dp = (1 - gamma - delta) * (alpha / (1 - alpha)) * (1 / np.log(2)) * (h_diff / h)

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

# -----------------------
# Key rate function
# -----------------------
def keyrate1(aldelt, gamma, n, qberthresh, esound, n_p):

    sol_val, sol, status = hterm1(n, aldelt, gamma, qberthresh, n_p)

    lambdaEC = 1.1 * (1 - gamma) * binh(qberthresh)
    epsEV = (aldelt * esound) / (1 + 2 * aldelt)
    epsPA = esound - epsEV

    rate = (
        sol_val
        - lambdaEC
        - (1 / n)
        * (
            np.log2(1 / epsEV)
            + (1 + aldelt) / aldelt * np.log2(1 / epsPA)
            - 2
        )
    )
    return rate, status

def rate_sixstate1(n, qberthresh, esound, n_p, n_aldelt, n_gamma):
    """
    Function computes the finite-size secure key rate for the entanglement-based BB84 protocol.
    It combines the worst-case entropy bound derived from 'htermBB84' with classical error-correction leakage and finite-sized security penalities to produce the final secure keyrate per signal as per Eqn 145 of arXiv:2405.05912.

    Parameters:
        aldelt: Finite-size security parameter related to the Renyi entropy order. Controls the tradeoff between keyrate and security confidence.
        gamma: Fraction of signals used for the channel rather than generating the final key.
        n: Total number of signals exchanged in the protocol.
        qberthresh: Maximum error rate that the protocol is willing to tolerate before the generated key is considered insecure.
        epsEV: Failure probability associated with error verification.
        epsPA: Failure probability associated with privacy amplification.
        n_p: Number of tangent points used to approximate nonlinear entropy-like functions.
    Returns:
        rate: Secure keyrate per signal under realistic conditions
        sol: The optimiser-selected values of the variables that achieved the worst-base bound.

    """

    gamma_range = 10 ** (-np.linspace(0, 2.5, n_gamma))
    aldelt_range = 10 ** (-np.linspace(np.log10(np.sqrt(n))-2, np.log10(np.sqrt(n))+2, n_aldelt))

    keyrate_partial = partial(keyrate1, n=n, qberthresh=qberthresh, esound=esound, n_p=n_p)

    opm_aldelt, opm_gamma, max_value = trialanderror(keyrate_partial, aldelt_range, gamma_range)

    return max_value, opm_aldelt, opm_gamma



# -----------------------
# Evaluating using the second Renyi Entropy Formula
# -----------------------

# -----------------------
# Convex optimization: multi-point linear underestimator
# -----------------------

def hterm2(n, aldelt, gamma, qberthresh, n_p):
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

    alpha = aldelt + 1 #convert back to alpha
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
    p_grid     = np.linspace(1e-12, 2/3 - 1e-12, n_p)   # perr

    # ------------------------
    # Base constraints
    # ------------------------
    constraints = [
        v1 >= 1e-12,
        v2 >= 1e-12,
        v2 <= qberthresh,
        perr >= 1e-12,
        perr <= 2/3 - 1e-12,
        1 - gamma*v1 - gamma*v2 >= 1e-12
    ]

    # ------------------------
    # Tangent Lines
    # ------------------------
    for pi in p_grid:
        # Value at reference point
        h = h = (
        2*pi
        + 2**(2 - alpha)
        * (
        (pi/2)**(1/alpha)
        + (1 - 3*pi/2)**(1/alpha)
        )**alpha
        )

        f_pi = (1 - gamma - delta) * ((np.log2(h))/(1 - alpha) + 1/(alpha - 1))

        # Gradient at that point
        h_diff = (
        2
        + 2**(1 - alpha)
        * (
        (pi/2)**(1/alpha)
        + (1 - 3*pi/2)**(1/alpha)
        )**(alpha - 1)
        * (
        (pi/2)**((1/alpha) - 1)
        - 3 * (1 - 3*pi/2)**((1/alpha) - 1)
        )
        )

        df_dp = (1 - gamma - delta) * (1 / (1 - alpha)) * (1 / np.log(2)) * (h_diff / h)

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

# -----------------------
# Key rate function
# -----------------------
def keyrate2(aldelt, gamma, n, qberthresh, esound, n_p):

    sol_val, sol, status = hterm2(n, aldelt, gamma, qberthresh, n_p)

    lambdaEC = 1.1 * (1 - gamma) * binh(qberthresh)
    epsEV = (aldelt * esound) / (1 + 2 * aldelt)
    epsPA = esound - epsEV

    rate = (
        sol_val
        - lambdaEC
        - (1 / n)
        * (
            np.log2(1 / epsEV)
            + (1 + aldelt) / aldelt * np.log2(1 / epsPA)
            - 2
        )
    )
    return rate, status

def rate_sixstate2(n, qberthresh, esound, n_p, n_aldelt, n_gamma):
    """
    Function computes the finite-size secure key rate for the entanglement-based BB84 protocol.
    It combines the worst-case entropy bound derived from 'htermBB84' with classical error-correction leakage and finite-sized security penalities to produce the final secure keyrate per signal as per Eqn 145 of arXiv:2405.05912.

    Parameters:
        aldelt: Finite-size security parameter related to the Renyi entropy order. Controls the tradeoff between keyrate and security confidence.
        gamma: Fraction of signals used for the channel rather than generating the final key.
        n: Total number of signals exchanged in the protocol.
        qberthresh: Maximum error rate that the protocol is willing to tolerate before the generated key is considered insecure.
        epsEV: Failure probability associated with error verification.
        epsPA: Failure probability associated with privacy amplification.
        n_p: Number of tangent points used to approximate nonlinear entropy-like functions.
    Returns:
        rate: Secure keyrate per signal under realistic conditions
        sol: The optimiser-selected values of the variables that achieved the worst-base bound.

    """

    gamma_range = 10 ** (-np.linspace(0, 2.5, n_gamma))
    aldelt_range = 10 ** (-np.linspace(np.log10(np.sqrt(n))-2, np.log10(np.sqrt(n))+2, n_aldelt))

    keyrate_partial = partial(keyrate2, n=n, qberthresh=qberthresh, esound=esound, n_p=n_p)

    opm_aldelt, opm_gamma, max_value = trialanderror(keyrate_partial, aldelt_range, gamma_range)

    return max_value, opm_aldelt, opm_gamma