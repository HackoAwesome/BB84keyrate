import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Step 1: Load the first CSV file
file1 = Path(__file__).parent.parent / "Data" / "results.csv"
data1 = pd.read_csv(file1)

# Step 1: Load the second CSV file
file2 = Path(__file__).parent.parent / "Data" / "results2.csv"
data2 = pd.read_csv(file2)

n_reference = np.array([10**j for j in range(3, 9)])
rates_reference = [0.0302146, 0.383833, 0.52904, 0.592656, 0.621459, 0.634525]

# Step 2: Plot the data
plt.plot(data1["n"], data1["rate"], marker='o', color='r', label="Original Parameterization")
plt.plot(data2["n"], data2["rate"], marker='o', color='b', label="Improved Parameterization")

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