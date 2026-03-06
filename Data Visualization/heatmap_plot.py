import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "/Users/junhui/Desktop/SP3172/BB84keyrate/Data/heatmap_result(100x100)(8).csv"

# Load CSV with first column as index
data = pd.read_csv(file, index_col=0)

# Extract ranges
gamma_range = data.index.astype(float).to_numpy()
aldelt_range = data.columns.astype(float).to_numpy()

# Extract heatmap values
grid = data.to_numpy()

plt.figure(figsize=(7,5))

plt.imshow(
    grid,
    origin='lower',
    aspect='auto',
    extent=[aldelt_range.min(), aldelt_range.max(),
            gamma_range.min(), gamma_range.max()]
)

plt.colorbar(label="Keyrate")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Alpha Delta")
plt.ylabel("Gamma")
plt.title("Keyrate Heatmap")

plt.show()