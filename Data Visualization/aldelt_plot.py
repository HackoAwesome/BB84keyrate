import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: Load the CSV file
file = Path(__file__).parent.parent / "Data" / "results_new(100x100).csv"
data = pd.read_csv(file)

x = data["n"].values
y = data["aldelt"].values  # alpha values

# Step 2: Compute error bars based on grid resolution
n_aldelt = 100
alpha_step = 4 / (n_aldelt - 1)
factor = 10**alpha_step

y_err_lower = y - (y / factor)
y_err_upper = (y * factor) - y
y_err = [y_err_lower, y_err_upper]

# Step 3: Linear regression in log-log space
logx = np.log10(x)
logy = np.log10(y)

slope, intercept = np.polyfit(logx, logy, 1)
logy_pred = slope * logx + intercept

# Step 4: Compute R^2
ss_res = np.sum((logy - logy_pred)**2)
ss_tot = np.sum((logy - np.mean(logy))**2)
r_squared = 1 - (ss_res / ss_tot)

# Convert back to power law
a = 10**intercept

print("Slope:", slope)
print("Intercept:", intercept)
print("R^2:", r_squared)
print(f"Fit: y = {a} * n^{slope}")

# Step 5: Generate fitted curve
x_fit = np.linspace(min(x)-500, max(x)+10**8, 200)
x_fit_sorted = np.sort(x_fit)
y_fit_sorted = a * x_fit_sorted**slope

# Step 6: Plot data with error bars
plt.figure(figsize=(7,5))
plt.errorbar(x, y, yerr=y_err,
             fmt='o', markersize=4,
             color='red', ecolor='gray',
             capsize=3, label="Data")

# Plot regression line
plt.plot(x_fit_sorted, y_fit_sorted, color='blue', label="Fit")

# Step 7: Set log scales
plt.xscale('log')
plt.yscale('log')

# Step 8: Labels, title, and grids
plt.xlabel('n')
plt.ylabel(r'$\Delta \alpha$')
plt.title(r"Optimal $\Delta \alpha$ against n")

# Major + minor grids
plt.grid(which='major', linestyle='-', linewidth=0.7, alpha=0.7)
plt.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)
plt.minorticks_on()

# Step 9: Add equation text (top-right)
eq_text = (
    r"$\log_{{10}}(\Delta \alpha) = {:.3f}\,\log_{{10}}(n) + {:.3f}$".format(slope, intercept)
    + "\n" +
    r"$R^2 = {:.4f}$".format(r_squared)
)

plt.text(0.95, 0.95, eq_text,
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

# Step 4: Show the graph
plt.savefig("alpha.svg", format="svg", bbox_inches="tight")
plt.show()