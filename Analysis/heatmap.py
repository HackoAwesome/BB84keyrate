import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pandas as pd

import sys
sys.path.append("../BB84keyrate/scripts")
from BB84keyrate import keyrate

n = 10**7
n_p = 100
qberthresh = 0.025
esound = 1e-10
grid_n = 200
file_path = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/heatmap_result.csv"

gamma_range = 10 ** (-np.linspace(0, 2.5, grid_n))
aldelt_range = 10 ** (-np.linspace(np.log10(np.sqrt(n))-2, np.log10(np.sqrt(n))+2, grid_n))

grid = np.zeros((grid_n, grid_n))

#The points where the convex optimisation failed to run is labelled as -2
#The points where the code returns infinity or the solution is not optimal is labelled as -1

for i, gamma in enumerate(gamma_range):
    for j, aldelt in enumerate(aldelt_range):
        try:
            value, status = keyrate(aldelt, gamma, n, qberthresh, esound, n_p)
            if not np.isfinite(value) or status != "optimal":
                value = 1
            grid[i, j] = value
        except Exception:
            grid[i, j] = 2

grid = np.array(grid)

df = pd.DataFrame(grid, index=gamma_range, columns=aldelt_range)
df.index.name = "gamma"
df.columns.name = "alpha_delta"

df.to_csv(file_path)

plt.figure(figsize=(7,5))

plt.imshow(
    grid,
    origin='lower',
    aspect='auto',
    extent=[aldelt_range.min(), aldelt_range.max(),
            gamma_range.min(), gamma_range.max()],
    interpolation='nearest'
)

plt.colorbar(label="Keyrate")

# Use log scale for axes
plt.xscale("log")
plt.yscale("log")

plt.xlabel("Alpha Delta")
plt.ylabel("gamma")
plt.title("Keyrate Heatmap")

plt.show()