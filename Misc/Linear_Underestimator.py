import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Parameters (pick representative values)
# -----------------------
gamma   = 0.1
hatdelt = 0.05
bdelt   = hatdelt / (1 + hatdelt)
n_ref_p = 100    #number of tangent point for p

# Domain
p = np.linspace(1e-6, 1, 500)

# -----------------------
# Original function
# -----------------------
def f(p):
    h = (1 - p)**(1 - bdelt) + p**(1 - bdelt)
    return (1 - gamma) * (1 - (1 / bdelt) * np.log2(h))

# -----------------------
# Derivative
# -----------------------
def df_dp(p):
    h = (1 - p)**(1 - bdelt) + p**(1 - bdelt)
    return (1 - gamma) * (
        -1 / (bdelt * np.log(2))
    ) * (
        (1 - bdelt) * (p**(-bdelt) - (1 - p)**(-bdelt))
    ) / h

# -----------------------
# Tangent points
# -----------------------
p_grid = np.linspace(0.1, 0.9, n_ref_p)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8, 5))

# Original curve
plt.plot(p, f(p), 'k', linewidth=2, label="Original f(p)")

# Tangent lines
for pi in p_grid:
    tangent = f(pi) + df_dp(pi) * (p - pi)
    plt.plot(p, tangent, '--', linewidth=1)

# Mark tangent points
#plt.scatter(p_grid, f(p_grid), color='red', zorder=5, label="Reference Points", s=10)

plt.xlabel("p")
plt.ylabel("f(p)")
plt.title("Original Function and Linear (Tangent) Underestimators")
plt.ylim(-0.2, 1)
#plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/hayleylim/Desktop/SP3172/100tangentlines.svg")