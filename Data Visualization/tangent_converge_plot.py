import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the CSV file
file = "/Users/hayleylim/Documents/GitHub/BB84keyrate/Data/tangent_converge_result(40x40)(8)new.csv"
data = pd.read_csv(file)


n = "$10^8$"

# Step 2: Plot the data
plt.plot(data["number of tangent"][7:], data["rate"][7:], marker='o', color='b', markersize=3)

# Step 3: Add labels and title
plt.legend()
plt.xlabel(r'Number of Tangent Lines ($n_{\text{tangent}}$)')
plt.ylabel('Keyrate')
plt.grid(True)
plt.title(f"Keyrate for {n} rounds")

# Step 4: Show the graph
plt.savefig("/Users/hayleylim/Desktop/SP3172/tangent(40x40).svg")
plt.show()