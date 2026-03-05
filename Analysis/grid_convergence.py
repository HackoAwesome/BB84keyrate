import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd

import sys
sys.path.append("../BB84keyrate/scripts")
from BB84keyrate import rateBB84

# -----------------------
# Compute data points
# -----------------------
def compute_datapoints(n_range, qberthresh, esound):
    """
    Computes secure keyrate data points over a range of block sizes.
    """

    rate_list = []

    for n in n_range:
        rate, opm_aldelt, opm_gamma, = rateBB84(nval, qberthresh, esound, n_ref_p, n, n)
        rate_list.append(rate)
        print(f"Computation completed for {n}x{n} grid size")
    
    data = pd.DataFrame({
        "size of grid": n_range,
        "rate": rate_list
    })

    return data

# -----------------------
# Parameters
# -----------------------
nval = 10**6
qberthresh = 0.025
esound = 1e-10
n_ref_p = 100     #number of tangent lines for underestimator
n_range = np.arange(0, 40, 5)  #size of grid (nxn)
file_path = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/grid_converge_result.csv"
# -----------------------
# Run computation
# -----------------------
datapts = compute_datapoints(n_range, qberthresh, esound)

# -----------------------
# Saving Data
# -----------------------
datapts.to_csv(file_path, index=False)

# -----------------------
# Plotting Results
# -----------------------
n_vals = datapts["size of grid"]
rates  = datapts["rate"]

plt.figure()
plt.plot(n_vals, rates, marker='o', color='b', label="new")
plt.legend()
plt.xlabel('Size of Grid')
plt.ylabel('Key Rate')
plt.grid(True)
plt.show()