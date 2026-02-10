import numpy as np

def trialanderror(func, grid_x, grid_y):
    best_value = -np.inf
    best_x = None
    best_y = None

    for xi in grid_x:
        for yi in grid_y:
            try:
                val = func(xi, yi)
            except Exception:
                # Skip points where the function cannot be evaluated
                continue

            if np.isfinite(val) and val > best_value:
                best_value = val
                best_x = xi
                best_y = yi

    return best_x, best_y, best_value
