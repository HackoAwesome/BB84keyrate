import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
file = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/results(100x100).csv"
data = pd.read_csv(file)

# Step 2: Plot the data
plt.plot(data["n"], data["aldelt"], marker='o', color='r')

# Step 3: Add labels and title
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('n')
plt.ylabel('alpha delta')
plt.grid(True)
plt.title("Optimal alpha delta against n")

# Step 4: Show the graph
plt.show()