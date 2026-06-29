import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Step 1: Load the CSV file
file = Path(__file__).parent.parent / "Data" / "results_new(100x100).csv"
data = pd.read_csv(file)

n_reference = np.array([10**j for j in range(3, 9)])
rates_reference = [0.0302146, 0.383833, 0.52904, 0.592656, 0.621459, 0.634525]

# Step 2: Plot the data
plt.plot(n_reference, rates_reference, marker='o', color='r', label="Original (Computed by Arqand et al.)")
plt.plot(data["n"], data["rate"], marker='o', color='b', label="New (With Convex Optimisation)")

# Step 3: Add labels and title
plt.legend(fontsize='small')
plt.xscale('log')
plt.ylim(0, 0.7)
plt.xlabel('Number of rounds (n)')
plt.ylabel('Keyrate')
plt.grid(True)
plt.title("Keyrate against n")

# Step 4: Show the graph
plt.show()