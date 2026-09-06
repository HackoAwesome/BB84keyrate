import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path

# Step 1: Load the first CSV file
file1 = Path(__file__).parent.parent / "Data" / "results_sixstate(100x100).csv"
data1 = pd.read_csv(file1)

# Step 1: Load the second CSV file
file2 = Path(__file__).parent.parent / "Data" / "results_new(100x100).csv"
data2 = pd.read_csv(file2)

n_reference = np.array([10**j for j in range(3, 9)])
rates_reference = [0.0302146, 0.383833, 0.52904, 0.592656, 0.621459, 0.634525]

fig, ax = plt.subplots()

# Step 2: Plot the data
ax.plot(data1["n"], data1["rate"], marker='o', color='b', label="Six-State Protocol")
#ax.plot(data2["n"], data2["rate"], marker='o', color='g', label="Improved Parameterization")

#ax.axhline(y=0.6627, linestyle="--", label = "Asymptotic rate", color = "black") #bb84 asymptotic rate
ax.axhline(y=0.70985, linestyle="--", label = "Asymptotic rate", color = "black") #sixstate asymptotic rate

# Rectangle: (x_start, y_start), width, height
rect = Rectangle(
    (10**2, 0),
    10**4 - 10**2,
    0.75,
    facecolor='red',
    alpha = 0.4
)

ax.add_patch(rect)

# Step 3: Add labels and title
plt.legend(fontsize='small', loc="lower right")
plt.xscale('log')
plt.ylim(0, 0.75)
plt.xlim(0.5*10**3, 0.2*10**9)
plt.xlabel('Number of rounds (n)')
plt.ylabel('Keyrate')
plt.grid(True)
plt.title("Keyrate against n")

# Step 4: Show the graph
plt.savefig("keyrate_plot3.svg", format="svg", bbox_inches="tight")
plt.show()