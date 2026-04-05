import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
file = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/results(100x100).csv"
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

# Step 4: Plot data with error bars ONLY (no separate scatter)
#plt.figure(figsize=(6,5))

plt.errorbar(x, y, yerr=y_err, fmt='o',
             color='red', ecolor='gray',
             elinewidth=1, capsize=3, markersize=4,
             label="Data")

# Sort x_fit for clean line
plt.plot(x_fit, y_fit, label=f"Fit: y = {a:.2e} n^{slope:.3f}\n$R^2$ = {r_squared:.4f}")

# Axes settings
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel(r'$\gamma$')
plt.title(r"Optimal $\gamma$ against n")
plt.grid()

#plt.legend()
plt.show()