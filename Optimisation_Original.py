import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# -----------------------
# Basic functions
# -----------------------

def log2(x):
    return np.log2(x)

def binh(p):
    return -p * log2(p) - (1 - p) * log2(1 - p)

# -----------------------
# Relative-entropy term
# q = (gamma*v1, gamma*v2, v3)
# nu_C = (gamma*(1-perr), gamma*perr, 1-gamma)
# -----------------------

def divtermBB84(v, perr, gamma):
    v1, v2, v3 = v
    return (
        gamma * v1 * log2(v1 / (1 - perr))
        + gamma * v2 * log2(v2 / perr)
        + v3 * log2(v3 / (1 - gamma))
    )

# -----------------------
# Convex optimization (via Differential Evolution)
# -----------------------

def htermBB84(hatdelt, gamma, qberthresh):

    bdelt = hatdelt / (1 + hatdelt)

    def objective(x):
        v1, v2, perr = x
        v3 = 1 - gamma * v1 - gamma * v2

        # Enforce positivity manually (Mathematica Piecewise analogue)
        if min(v1, v2, v3, perr, 1 - perr) <= 0:
            return 1e5

        term1 = (1 - gamma * v1 - gamma * v2)
        term2 = 1 - (1 / bdelt) * log2(
            (1 - perr) ** (1 - bdelt) + perr ** (1 - bdelt)
        )
        term3 = (1 / hatdelt) * divtermBB84(
            [v1, v2, v3], perr, gamma
        )

        return term1 * term2 + term3

    bounds = [
        (1e-12, 1),                 # v1
        (1e-12, qberthresh),        # v2 < qberthresh
        (1e-12, 1 - 1e-12)          # 0 < perr < 1
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        polish=True
    )

    return result.fun, result

# -----------------------
# Final key rate
# -----------------------

def rateBB84(aldelt, gamma, n, qberthresh, epsEV, epsPA):

    hatdelt = aldelt / (1 - aldelt)

    sol_val, sol = htermBB84(hatdelt, gamma, qberthresh)

    lambdaEC = 1.1 * (1 - gamma) * binh(qberthresh)

    rate = (
        sol_val
        - lambdaEC
        - (1 / n)
        * (
            log2(1 / epsEV)
            + (1 + aldelt) / aldelt * log2(1 / epsPA)
            - 2
        )
    )

    return rate, sol

# -----------------------
# Set up n, alpha, gamma values
# -----------------------

nvals = np.array([10**j for j in range(3, 9)])

aldeltvals = 10 ** (-np.array([0.7, 1.4, 2.1, 2.7, 3.4, 4.0]))
gammavals  = 10 ** (-np.array([0.4, 0.8, 1.2, 1.5, 1.9, 2.2]))

# -----------------------
# Compute data points
# -----------------------

def compute_datapoints():
    qberthresh = 0.025
    esound = 1e-10

    datapts = []

    for ptnum in range(6):
        # Extract parameters
        n = nvals[ptnum]
        aldelt = aldeltvals[ptnum]
        gamma = gammavals[ptnum]

        # Optimal epsEV and epsPA
        epsEV = (aldelt * esound) / (1 + 2 * aldelt)
        epsPA = esound - epsEV

        # Compute rate
        rate, sol = rateBB84(
            aldelt,
            gamma,
            n,
            qberthresh,
            epsEV,
            epsPA
        )

        # Store (n, rate)
        datapts.append((n, rate))

    return np.array(datapts)


# Run computation
datapts = compute_datapoints()

# datapts assumed to be an array of shape (N, 2)
# column 0: n values
# column 1: rates

n_vals = datapts[:, 0]
rates  = datapts[:, 1]

plt.figure()
plt.plot(n_vals, rates, marker='o')
plt.xscale('log')              # log scale on x-axis
plt.ylim(0, 0.7)               # PlotRange -> {0, .7}
plt.xlabel('n')
plt.ylabel('Key rate')
plt.grid(True)

plt.show()