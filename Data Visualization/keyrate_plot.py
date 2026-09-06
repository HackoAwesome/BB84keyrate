import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path

# Step 1: Load the CSV file
file = Path(__file__).parent.parent / "Data" / "results(100x100).csv"
data = pd.read_csv(file)

n_reference = np.array([10**j for j in range(3, 9)])
rates_reference = [0.0302146, 0.383833, 0.52904, 0.592656, 0.621459, 0.634525]

fig, ax = plt.subplots()

# Step 2: Plot the data
ax.plot(n_reference, rates_reference, marker='o', color='purple', label="Original (Computed by Arqand et al.)")
ax.plot(data["n"], data["rate"], marker='o', color='b', label="New (With Convex Optimisation)")

plt.axhline(y=0.6627, linestyle="--", label = "Asymptotic rate", color = "black") #bb84 asymptotic rate
#plt.axhline(y=0.70985, linestyle="--", label = "Asymptotic rate", color = "black") #sixstate asymptotic rate

# Rectangle: (x_start, y_start), width, height
rect = Rectangle(
    (10**2, 0),
    10**4 - 10**2,
    0.7,
    facecolor='red',
    alpha = 0.4
)

ax.add_patch(rect)

# Step 3: Add labels and title
plt.legend(fontsize='small')
plt.xscale('log')
plt.ylim(0, 0.7)
plt.xlim(0.5*10**3, 0.2*10**9)
plt.xlabel('Number of rounds (n)')
plt.ylabel('Keyrate')
plt.grid(True)
plt.title("Keyrate against n")

# Step 4: Show the graph
plt.savefig("keyrate_plot.svg", format="svg", bbox_inches="tight")
plt.show()