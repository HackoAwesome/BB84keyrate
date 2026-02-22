import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
file = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/results(100x100).csv"
data = pd.read_csv(file)

rates_reference = [0.0302146, 0.383833, 0.52904, 0.592656, 0.621459, 0.634525]

# Step 2: Plot the data
plt.plot(data["n"], rates_reference, marker='o', color='b', label="original")
plt.plot(data["n"], data["rate"], marker='o', color='r', label="new")

# Step 3: Add labels and title
plt.legend()
plt.xscale('log')
plt.ylim(0, 0.7)
plt.xlabel('n')
plt.ylabel('Key Rate')
plt.grid(True)
plt.title("Key Rate against n")

# Step 4: Show the graph
plt.show()