import pandas as pd

# Path to your file
file = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/heatmap_result(200x200)(8).csv"

# Load CSV
data = pd.read_csv(file, index_col=0)

# Replace values
data.replace({-1: 1, -2: 2}, inplace=True)

# Save back to the same file
data.to_csv(file)

print("File successfully updated.")