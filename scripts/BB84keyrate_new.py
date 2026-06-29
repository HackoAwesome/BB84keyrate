import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd

from entropic_terms_new import binh, hterm
from general_optimisation import trialanderror

# -----------------------
# Key rate function
# -----------------------
def keyrate(aldelt, gamma, n, qberthresh, esound, n_p):

    sol_val, sol, status = hterm(n, aldelt, gamma, qberthresh, n_p)

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

def rateBB84(n, qberthresh, esound, n_p, n_aldelt, n_gamma):
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

    keyrate_partial = partial(keyrate, n=n, qberthresh=qberthresh, esound=esound, n_p=n_p)

    opm_aldelt, opm_gamma, max_value = trialanderror(keyrate_partial, aldelt_range, gamma_range)

    return max_value, opm_aldelt, opm_gamma
