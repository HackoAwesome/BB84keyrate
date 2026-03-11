import numpy as np

def trialanderror(func, grid_x, grid_y):
    best_value = -np.inf
    best_x = None
    best_y = None

    print("Grid Search has started.")
    progress = 0
    total_progress = len(grid_x)

    for xi in grid_x:
        progress += 1
        for yi in grid_y:
            try:
                val, status = func(xi, yi)
                if status == "optimal":
                    val2 = val
            except Exception:
                # Skip points where the function cannot be evaluated
                continue
            if np.isfinite(val2) and val2 > best_value:
                best_value = val2
                best_x = xi
                best_y = yi
        
        #Update progress
        print(f"Grid search is {progress/total_progress*100:.2f}% complete.")

    print("Optimal parameters have been found.")
    return best_x, best_y, best_value
