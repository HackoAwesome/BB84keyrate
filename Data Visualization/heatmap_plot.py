import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

file = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/heatmap_result(100x100)(8).csv"

# Load CSV with first column as index
data = pd.read_csv(file, index_col=0)

# Extract ranges
gamma_range = data.index.astype(float).to_numpy()
aldelt_range = data.columns.astype(float).to_numpy()

# Extract heatmap values
grid = data.to_numpy()

# Maximum and minimum values for normal data (exclude -1)
normal_values = grid[grid != -1]
vmin = np.min(normal_values)
vmax = np.max(normal_values)

# Create a colormap:
# - First color = red for infinite points (-1)
# - Remaining colors = viridis for all other values
viridis_colors = plt.cm.viridis(np.linspace(0, 1, 256))
colors = np.vstack((
    [[1,0,0,1]],  # red for infinite points
    viridis_colors
))
custom_cmap = ListedColormap(colors)

# Define boundaries: -1 maps to first color, rest maps to viridis
bounds = [-1.01, -0.99] + list(np.linspace(vmin, vmax, 256))
norm = BoundaryNorm(bounds, custom_cmap.N)

# Compute cell edges for pcolormesh (log scale)
x_edges = np.logspace(np.log10(aldelt_range[0]), np.log10(aldelt_range[-1]), len(aldelt_range)+1)
y_edges = np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[-1]), len(gamma_range)+1)

plt.figure(figsize=(7,5))

# Plot heatmap
c = plt.pcolormesh(
    x_edges,
    y_edges,
    grid,
    cmap=custom_cmap,
    norm=norm,
    shading='auto'
)

# Mark the maximum keyrate point (ignore -1)
# Find indices of max value
max_idx = np.unravel_index(np.argmax(grid * (grid != -1)), grid.shape)
max_gamma = gamma_range[max_idx[0]]
max_aldelt = aldelt_range[max_idx[1]]

# Overlay the max point
plt.scatter(
    max_aldelt,
    max_gamma,
    color='white',
    marker='*',
    s=150,
    edgecolors='black',
    label='Max Keyrate'
)

plt.xscale("log")
plt.yscale("log")
plt.colorbar(c, label="Keyrate")
plt.xlabel("Alpha Delta")
plt.ylabel("Gamma")
plt.title("Keyrate Heatmap for 10^8 rounds")
plt.legend()
plt.show()