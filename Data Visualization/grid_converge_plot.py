import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
file = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/grid_converge_result(6).csv"
data = pd.read_csv(file)

n = "10^6"

# Step 2: Plot the data
plt.plot(data["size of grid"], data["rate"], marker='o', color='b')

# Step 3: Add labels and title
plt.legend()
plt.xlabel('Size of Grid(nxn)')
plt.ylabel('Key Rate')
plt.grid(True)
plt.title(f"Key Rate for {n} rounds")

# Step 4: Show the graph
plt.show()