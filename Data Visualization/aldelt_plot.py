import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
file = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/results(100x100).csv"
data = pd.read_csv(file)

y = data["aldelt"].values
x = data["n"].values

# alpha values
y = data["aldelt"].values

# grid resolution error (from correct range)
n_aldelt = 100
alpha_step = 4 / (n_aldelt - 1)
factor = 10**alpha_step

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
plt.errorbar(x, y, yerr=y_err,
             fmt='o', markersize=4,
             color='red', ecolor='gray',
             capsize=3, label="Data")


# Plot regression
plt.plot(x_fit, y_fit, label=f"Fit: y={a:.2e} n^{slope:.3f}")

# Step 3: Add labels and title
#plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel(r'$\Delta \alpha$')
plt.grid(True)
plt.title(r"Optimal $\Delta \alpha$ against n")

# Step 4: Show the graph
plt.show()