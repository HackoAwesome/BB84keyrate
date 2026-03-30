import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the CSV file
file = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/results(100x100).csv"
data = pd.read_csv(file)

n_reference = np.array([10**j for j in range(3, 9)])
rates_reference = np.array([0.0302146, 0.383833, 0.52904, 0.592656, 0.621459, 0.634525])

original = np.array([0.00926586, 0.37584893, 0.52710025, 0.59213808, 0.62123808, 0.63412847])

print(rates_reference-data['rate'])

plt.plot(n_reference, data["rate"]-original, marker='o', color='b')

plt.legend()
plt.xscale('log')
plt.xlabel('Number of rounds')
plt.ylabel('Difference in Keyrate')
plt.grid(True)
plt.title("Difference in Keyrate against n")
plt.show()