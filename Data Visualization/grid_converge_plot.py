import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
file = "/Users/hayleylim/Documents/GitHub/BB84keyrate/Data/grid_converge_result(8).csv"
data = pd.read_csv(file)

n = "$10^8$"

# Step 2: Plot the data
plt.plot(data["size of grid"], data["rate"], marker='o', color='b')

# Step 3: Add labels and title
plt.legend()
plt.xlabel(r'Size of Grid(n $\times$ n)')
plt.ylabel('Keyrate')
plt.grid(True)
plt.title(f"Keyrate for {n} rounds")

# Step 4: Show the graph
plt.savefig("/Users/hayleylim/Desktop/SP3172/gridconverge.svg")
plt.show()