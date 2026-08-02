import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: Load the CSV file
file = Path(__file__).parent.parent / "Data" / "results.csv"
data = pd.read_csv(file)

y = data["gamma"].values
x = data["n"].values

#error in grid search
log_min = -2.5
log_max = 0
num_points = 100

delta = (log_max - log_min) / (num_points - 1)
factor = 10**delta

y_err_lower = y - (y / factor)
y_err_upper = (y * factor) - y

y_err = [y_err_lower, y_err_upper]

# Step 2: Perform regression in log space
logx = np.log10(x)
logy = np.log10(y)

# Step 3: Linear regression
slope, intercept = np.polyfit(logx, logy, 1)

# Predicted log values
logy_pred = slope * logx + intercept

# Step 4: Compute R^2
ss_res = np.sum((logy - logy_pred)**2)
ss_tot = np.sum((logy - np.mean(logy))**2)
r_squared = 1 - (ss_res / ss_tot)

print("Slope:", slope)
print("Intercept:", intercept)
print("R^2:", r_squared)

# Convert back to power law
a = 10**intercept
print(f"Fit: y = {a} * n^{slope}")

# Step 3: Generate fitted curve
x_fit = np.linspace(min(x)-500, max(x)+10**8, 200)
y_fit = a * x_fit**slope

# Step 4: Plot data
#plt.scatter(x, y, marker='o', color='r', label="Data")

# Plot data with error bars
plt.figure(figsize=(6,5))

plt.errorbar(x, y, yerr=y_err, fmt='o',
             color='red', ecolor='gray',
             elinewidth=1, capsize=3, markersize=4,
             label="Data")

# Sort x_fit for clean plotting
x_fit_sorted = np.sort(x_fit)
y_fit_sorted = a * x_fit_sorted**slope

# Plot fitted curve
plt.plot(x_fit_sorted, y_fit_sorted, color='blue')

# Log scales
plt.xscale('log')
plt.yscale('log')

# Labels and title
plt.xlabel('n')
plt.ylabel(r'$\gamma$')
plt.title(r"Optimal $\gamma$ against n")

# Major + minor gridlines
plt.grid(which='major', linestyle='-', linewidth=0.7, alpha=0.7)
plt.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)

# Enable minor ticks explicitly (important for log plots)
plt.minorticks_on()

# Add best-fit equation on plot
# Linear equation in log space
eq_text = (
    r"$\log_{{10}}(\gamma) = {:.3f}\,\log_{{10}}(n) + {:.3f}$".format(slope, intercept)
    + "\n" +
    r"$R^2 = {:.4f}$".format(r_squared)
)

# Place on top-right
plt.text(0.95, 0.95, eq_text,
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment='top',
         horizontalalignment='right',  # aligns the text box to the right
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

plt.show()