import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
from pathlib import Path

import sys
sys.path.append("../BB84keyrate/scripts")
from sixstate_keyrate import rate_sixstate1, rate_sixstate2, rate_sixstate3

# -----------------------
# Compute data points
# -----------------------
def compute_datapoints(n_range, qberthresh, esound, keyrate_function):
    """
    Computes secure keyrate data points over a range of block sizes.
    """

    gamma_list = []
    aldelt_list = []
    rate_list = []

    for n in n_range:
        rate, opm_aldelt, opm_gamma, = keyrate_function(n, qberthresh, esound, n_ref_p, n_sam_aldelt, n_sam_gamma)
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
file_path1 = Path(__file__).parent.parent / "Data" / "results.csv"
file_path2 = Path(__file__).parent.parent / "Data" / "results2.csv"

# -----------------------
# Run computation
# -----------------------
print("Computing the first formula")
datapts1 = compute_datapoints(nvals, qberthresh, esound, rate_sixstate1)
print("Computation completed for first formula")
print("Computing the formula formula")
datapts2 = compute_datapoints(nvals, qberthresh, esound, rate_sixstate3)
print("Computation completed for second formula")

# -----------------------
# Saving Data
# -----------------------
datapts1.to_csv(file_path1, index=False)
datapts2.to_csv(file_path2, index=False)

# -----------------------
# Plotting Results
# -----------------------
#First Renyi Entropy formula
n_vals1 = datapts1["n"]
rates1  = datapts1["rate"]

#Second Renyi Entropy formula
n_vals2 = datapts2["n"]
rates2  = datapts2["rate"]

n_reference = np.array([10**j for j in range(3, 9)])
rates_reference = [0.0302146, 0.383833, 0.52904, 0.592656, 0.621459, 0.634525]

plt.figure()
plt.plot(n_vals1, rates1, marker='o', color='g', label="First Formula")
plt.plot(n_vals2, rates2, marker='o', color='b', label="Second Formula")
plt.plot(n_reference, rates_reference, marker='o', color='r', label="original")
plt.legend()
plt.xscale('log')
plt.ylim(0, 0.7)
plt.xlabel('n')
plt.ylabel('Key Rate')
plt.grid(True)
plt.show()