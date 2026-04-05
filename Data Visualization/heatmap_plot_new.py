import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

file = "/Users/hayleylim/Documents/GitHub/BB84keyrate/Data/heatmap_result(200x200)(8)(relax).csv"

# Load CSV
data = pd.read_csv(file, index_col=0)

gamma_range = data.index.astype(float).to_numpy()
aldelt_range = data.columns.astype(float).to_numpy()
grid = data.to_numpy()
X, Y = np.meshgrid(aldelt_range, gamma_range)

# ---------------------------
# Separate values and status
# ---------------------------
not_optimal_mask = grid == 1
failed_mask = grid == 2

# Keyrate grid (replace invalid values with NaN)
keyrate = grid.astype(float)
keyrate[not_optimal_mask] = np.nan
keyrate[failed_mask] = np.nan


# ---------------------------
# Compute min/max safely
# ---------------------------
vmin = np.nanmin(keyrate)
vmax = np.nanmax(keyrate)

# ---------------------------
# Compute cell edges (log scale)
# ---------------------------
x_edges = np.logspace(np.log10(aldelt_range[0]),
                      np.log10(aldelt_range[-1]),
                      len(aldelt_range) + 1)

y_edges = np.logspace(np.log10(gamma_range[0]),
                      np.log10(gamma_range[-1]),
                      len(gamma_range) + 1)

plt.figure(figsize=(7,5))

# ---------------------------
# Plot main heatmap
# ---------------------------
c = plt.pcolormesh(
    x_edges,
    y_edges,
    keyrate,
    cmap="viridis",
    shading="auto",
    vmin=vmin,
    vmax=vmax
)


contours = plt.contour(
    X, Y, keyrate,
    levels=10,              # adjust (5–15 is good)
    colors='white',
    linewidths=0.8,
)

plt.clabel(contours, inline=True, fontsize=7)

# ---------------------------
# Overlay solver statuses
# ---------------------------
plt.pcolormesh(
    x_edges,
    y_edges,
    not_optimal_mask,
    cmap=ListedColormap(["none", "gray"]),
    shading="auto"
)

plt.pcolormesh(
    x_edges,
    y_edges,
    failed_mask,
    cmap=ListedColormap(["none", "gray"]),
    shading="auto"
)


# ---------------------------
# Find maximum keyrate
# ---------------------------
max_idx = np.unravel_index(np.nanargmax(keyrate), keyrate.shape)
print(keyrate.shape)

max_gamma = gamma_range[max_idx[0]]
max_aldelt = aldelt_range[max_idx[1]]

plt.scatter(
    max_aldelt,
    max_gamma,
    color="white",
    marker="*",
    s=150,
    edgecolors="black",
    label="Max Keyrate"
)

print(f"Aldelt range: {np.min(aldelt_range)} → {np.max(aldelt_range)}")


# ---------------------------
# Plot settings
# ---------------------------
plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$\Delta \alpha$")
plt.ylabel(r"$\gamma$")
plt.title("Keyrate Heatmap for $10^8$ rounds")

plt.colorbar(c, label="Keyrate")
plt.annotate(
    f"({max_aldelt:.2e}, {max_gamma:.2e})",
    (max_aldelt, max_gamma),
    textcoords="offset points",
    xytext=(10, 10),
    ha='left',
    color='black',
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
)

# Custom legend
legend_elements = [
    Patch(facecolor="gray", label="Optimization failed"),
]

plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), borderaxespad=0, handles=[*legend_elements, plt.Line2D([],[],marker="*",color="white",
           markeredgecolor="black",linestyle="",label="Max Keyrate")])

plt.show()