import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd

import sys
sys.path.append("../BB84keyrate/scripts")
from BB84keyrate_new import rateBB84

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
        print(f"Computation completed for n = {n}")
    
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
n_sam_gamma = 100  #number of sample points for gamma (original used 25)
n_sam_aldelt = 100 #number of sample points for aldelt (original used 40)
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
n_reference = np.array([10**j for j in range(3, 9)])
rates_reference = [0.0302146, 0.383833, 0.52904, 0.592656, 0.621459, 0.634525]

plt.figure()
plt.plot(n_vals, rates, marker='o', color='b', label="new")
plt.plot(n_reference, rates_reference, marker='o', color='r', label="original")
plt.legend()
plt.xscale('log')
plt.ylim(0, 0.7)
plt.xlabel('n')
plt.ylabel('Key Rate')
plt.grid(True)
plt.show()