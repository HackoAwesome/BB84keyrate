import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd

from entropic_terms import binh, hterm
from general_optimisation import trialanderror

# -----------------------
# Key rate function
# -----------------------
def keyrate(aldelt, gamma, n, qberthresh, esound, n_p):

    hatdelt = aldelt / (1 - aldelt)

    sol_val, sol = hterm(n, hatdelt, gamma, qberthresh, n_p)

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
    return rate

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

# -----------------------
# Compute data points
# -----------------------
def compute_datapoints(n_range, qberthresh, esound):
    """
    Computes secure keyrate data points over a range of block sizes.
    """

    gamma_list = []
    aldelt_list = []
    rate_list = []

    for n in n_range:
        rate, opm_aldelt, opm_gamma, = rateBB84(n, qberthresh, esound, n_ref_p, n_sam_aldelt, n_sam_gamma)
        gamma_list.append(opm_gamma)
        aldelt_list.append(opm_aldelt)
        rate_list.append(rate)
    
    data = pd.DataFrame({
        "n": n_range,
        "gamma": gamma_list,
        "aldelt": aldelt_list,
        "rate": rate_list
    })

    return data

# -----------------------
# Parameters
# -----------------------
nvals = np.array([10**j for j in range(3, 9)])
qberthresh = 0.025
esound = 1e-10
n_ref_p = 100     #number of tangent lines for underestimator
n_sam_gamma = 10  #number of sample points for gamma (minimally 25 to attain better results than original)
n_sam_aldelt = 10 #number of sample points for aldelt (minimally 40 to attain better results than original)
file_path = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/results.csv"
# -----------------------
# Run computation
# -----------------------
datapts = compute_datapoints(nvals, qberthresh, esound)

# -----------------------
# Saving Data
# -----------------------
datapts.to_csv(file_path, index=False)

# -----------------------
# Plotting Results
# -----------------------
n_vals = datapts["n"]
rates  = datapts["rate"]
rates_reference = [0.0302146, 0.383833, 0.52904, 0.592656, 0.621459, 0.634525]

plt.figure()
plt.plot(n_vals, rates, marker='o', color='b', label="new")
plt.plot(n_vals, rates_reference, marker='o', color='r', label="original")
plt.legend()
plt.xscale('log')
plt.ylim(0, 0.7)
plt.xlabel('n')
plt.ylabel('Key Rate')
plt.grid(True)
plt.show()