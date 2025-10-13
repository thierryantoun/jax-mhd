from itertools import product

def optimal_3d_partition(n_devices):
    """Return the best (dx, dy, dz) such that dx*dy*dz == n_devices and dimensions are as close as possible."""
    candidates = [(x, y)
                  for x, y in product(range(1, n_devices+1), repeat=2)
                  if x * y  == n_devices]
    return min(candidates, key=lambda t: max(t) - min(t))