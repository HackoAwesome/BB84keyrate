import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd
from pathlib import Path

import sys
sys.path.append("../BB84keyrate/scripts")
from BB84keyrate import rateBB84

# -----------------------
# Compute data points
# -----------------------
def compute_datapoints(p_range, qberthresh, esound):
    """
    Computes secure keyrate data points over a range of block sizes.
    """

    rate_list = []

    for p in p_range:
        rate, opm_aldelt, opm_gamma, = rateBB84(nval, qberthresh, esound, p, n_sam_aldelt, n_sam_gamma)
        rate_list.append(rate)
        print(f"Computation completed for {p} tangent lines")
    
    data = pd.DataFrame({
        "number of tangent": p_range,
        "rate": rate_list
    })

    return data

# -----------------------
# Parameters
# -----------------------
nval = 10**8
qberthresh = 0.025
esound = 1e-10
p_range = np.arange(0, 150, 2)     #number of tangent lines for underestimator
n_sam_gamma = 40  #number of sample points for gamma (minimally 25 to attain better results than original)
n_sam_aldelt = 40 #number of sample points for aldelt (minimally 40 to attain better results than original)
file_path = Path(__file__).parent.parent / "Data" / "results.csv"
# -----------------------
# Run computation
# -----------------------
datapts = compute_datapoints(p_range, qberthresh, esound)

# -----------------------
# Saving Data
# -----------------------
datapts.to_csv(file_path, index=False)

# -----------------------
# Plotting Results
# -----------------------
n_vals = datapts["number of tangent"]
rates  = datapts["rate"]

plt.figure()
plt.plot(n_vals, rates, marker='o', color='b')
plt.legend()
plt.xlabel('Number of Tangents')
plt.ylabel('Key Rate')
plt.grid(True)
plt.show()