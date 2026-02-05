import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# -----------------------
# Binary entropy (numeric)
# -----------------------
def binh(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

# -----------------------
# KL divergence term
# -----------------------
def divtermBB84(v1, v2, v3, perr, gamma):
    return (
        cp.kl_div(gamma*v1, gamma*(1-perr))
      + cp.kl_div(gamma*v2, gamma*perr)
      + cp.kl_div(v3, 1-gamma)
    ) / np.log(2)

# -----------------------
# Convex optimization: multi-point linear underestimator
# -----------------------
def htermBB84(hatdelt, gamma, qberthresh, n_v, n_p):
    bdelt = hatdelt / (1 + hatdelt)

    # CVXPY variables
    v1   = cp.Variable(nonneg=True)
    v2   = cp.Variable(nonneg=True)
    perr = cp.Variable()
    v3   = 1 - gamma*v1 - gamma*v2
    t    = cp.Variable()  # epigraph for the product term

    # ------------------------
    # Multi-point linear underestimator
    # ------------------------
    v_ref_grid = np.linspace(1e-12, gamma, n_v)
    p_ref_grid = np.linspace(1e-12, qberthresh, n_p)

    # ------------------------
    # Bounds
    # ------------------------
    constraints = [
        v1 >= 1e-12,
        v2 >= 1e-12,
        v2 <= qberthresh,
        perr >= 1e-12,
        perr <= 1 - 1e-12,
        v3 >= 1e-12,
    ]

    for vi in v_ref_grid:
        for pi in p_ref_grid:
            # Function value at reference point
            f0 = (1 - vi) * (1 - (1/bdelt) * np.log2((1-pi)**(1-bdelt) + pi**(1-bdelt)))

            # Partial derivatives
            df_dv = -(1 - (1/bdelt) * np.log2((1-pi)**(1-bdelt) + pi**(1-bdelt)))
            h = (1-pi)**(1-bdelt) + pi**(1-bdelt)
            df_dp = (1 - vi) * (-1/(bdelt*np.log(2))) * (-(1-bdelt)*(1-pi)**(-bdelt) + (1-bdelt)*pi**(-bdelt))/h

            # Linear underestimator only (no quadratic)
            t_constraint = f0 + df_dv*(gamma*v1 + gamma*v2 - vi) + df_dp*(perr - pi)
            constraints.append(t >= t_constraint)

    # ------------------------
    # KL divergence term
    # ------------------------
    term2 = (1/hatdelt)*divtermBB84(v1, v2, v3, perr, gamma)
    objective = cp.Minimize(t + term2)

    # ------------------------
    # Solve
    # ------------------------
    prob = cp.Problem(objective, constraints)
    prob.solve(
    solver=cp.SCS,
    max_iters=50000,
    verbose = True
    )



    return prob.value, (v1.value, v2.value, perr.value)

# -----------------------
# Key rate function
# -----------------------
def rateBB84(aldelt, gamma, n, qberthresh, epsEV, epsPA, n_v, n_p):
    hatdelt = aldelt / (1 - aldelt)

    sol_val, sol = htermBB84(hatdelt, gamma, qberthresh, n_v, n_p)

    lambdaEC = 1.1 * (1 - gamma) * binh(qberthresh)

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

    return rate, sol

# -----------------------
# Parameters
# -----------------------
nvals = np.array([10**j for j in range(3, 9)])
aldeltvals = 10 ** (-np.array([0.7, 1.4, 2.1, 2.7, 3.4, 4.0]))
gammavals  = 10 ** (-np.array([0.4, 0.8, 1.2, 1.5, 1.9, 2.2]))
n_ref_v = 9     #number of reference point for v
n_ref_p = 8     #number of reference point for p

# -----------------------
# Compute data points
# -----------------------
def compute_datapoints():
    qberthresh = 0.025
    esound = 1e-10
    datapts = []

    for ptnum in range(6):
        n = nvals[ptnum]
        aldelt = aldeltvals[ptnum]
        gamma = gammavals[ptnum]

        epsEV = (aldelt * esound) / (1 + 2 * aldelt)
        epsPA = esound - epsEV

        rate, sol = rateBB84(aldelt, gamma, n, qberthresh, epsEV, epsPA, n_ref_v, n_ref_p)
        datapts.append((n, rate))

    return np.array(datapts)

# -----------------------
# Run computation
# -----------------------
datapts = compute_datapoints()
n_vals = datapts[:, 0]
rates  = datapts[:, 1]
rates_reference = [0.0302146, 0.383833, 0.52904, 0.592656, 0.621459, 0.634525]

plt.figure()
plt.plot(n_vals, rates, marker='o', color='b', label="SCS")
plt.plot(n_vals, rates_reference, marker='o', color='r', label="DE")
plt.legend()
plt.xscale('log')
plt.ylim(0, 0.7)
plt.xlabel('n')
plt.ylabel('Key Rate')
plt.grid(True)
plt.show()