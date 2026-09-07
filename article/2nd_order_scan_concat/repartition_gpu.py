from itertools import product

def optimal_3d_partition(n_devices, Nx, Ny, Nz):
    """Return the best (dx, dy, dz) such that dx*dy*dz == n_devices, each grid
    dimension is evenly divisible by its partition factor (Nx % dx == 0, etc.),
    and the partition is as cubic as possible (dimensions as close as possible).
    """
    candidates = [(x, y, z)
                  for x, y, z in product(range(1, n_devices + 1), repeat=3)
                  if x * y * z == n_devices
                  and Nx % x == 0 and Ny % y == 0 and Nz % z == 0]

    if not candidates:
        raise ValueError(
            f"No valid 3D partition of {n_devices} device(s) evenly divides the "
            f"grid ({Nx}, {Ny}, {Nz}). Pick a device count or a resolution whose "
            f"factors are compatible (e.g. resolution multiples of the device "
            f"count on at least one axis)."
        )

    return min(candidates, key=lambda t: max(t) - min(t))