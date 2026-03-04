#!/usr/bin/env python3

import os
import time
import jax
from functools import partial
from pathlib import Path, PurePath
import json

from modules import *
from numerical_scheme import *
from initial_conditions import *
from repartition_gpu import *
from load_config import *
from physics import *

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding


def main(args, config):

    IC = config["simulation"]["IC"]
    Nx = int(config["simulation"]["resolution_x"])
    Ny = int(config["simulation"]["resolution_y"])
    Nz = int(config["simulation"]["resolution_z"])
    use_double = config.getboolean("simulation", "double")
    boxsize = float(config["simulation"]["boxsize"])
    gamma = float(config["simulation"]["gamma"])
    courant_fac = float(config["simulation"]["courant_fac"])
    t_stop = float(config["simulation"]["t_stop"])

    if use_double:
        print("Using double precision")
        jax.config.update("jax_enable_x64", True)
    else:
        print("Using single precision")

    dx = boxsize / Nx
    dy = boxsize / Ny
    dz = boxsize / Nz

    # Domain
    xlin = jnp.linspace(0.5 * dx, boxsize - 0.5 * dx, Nx)
    ylin = jnp.linspace(0.5 * dy, boxsize - 0.5 * dy, Ny)
    zlin = jnp.linspace(0.5 * dz, boxsize - 0.5 * dz, Nz)
    X, Y, Z = jnp.meshgrid(xlin, ylin, zlin, indexing="ij")

    n_devices = jax.device_count()
    x_opt, y_opt, z_opt = optimal_3d_partition(n_devices)
    mesh = Mesh(mesh_utils.create_device_mesh((x_opt, y_opt, z_opt)), ("x", "y", "z"))
    sharding = NamedSharding(mesh, PartitionSpec("x", "y", "z"))

    X = jax.lax.with_sharding_constraint(X, sharding)
    Y = jax.lax.with_sharding_constraint(Y, sharding)
    Z = jax.lax.with_sharding_constraint(Z, sharding)

    # Initial conditions
    rho, vx, vy, vz, Bx, By, Bz, P = inital_condition(IC, X, Y, Z, gamma, boxsize)
    Mass, Momx, Momy, Momz, Energy, Bx, By, Bz = get_conserved(rho, vx, vy, vz, P, gamma, Bx, By, Bz)

    # -------------------------
    # WARMUP (compile seulement)
    # -------------------------
    print("Warmup (JIT compilation)...")
    Mass, Momx, Momy, Momz, Energy, dt, Bx, By, Bz = update(
        Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac, Bx, By, Bz
    )
    jax.block_until_ready((Mass, Momx, Momy, Momz, Energy, Bx, By, Bz))

    # Reset initial conditions après le warmup
    rho, vx, vy, vz, Bx, By, Bz, P = inital_condition(IC, X, Y, Z, gamma, boxsize)
    Mass, Momx, Momy, Momz, Energy, Bx, By, Bz = get_conserved(rho, vx, vy, vz, P, gamma, Bx, By, Bz)

    # -------------------------
    # PROFILING
    # -------------------------
    global_start = time.time()
    t = 0.0
    n_iter = 0
    trace_done = False

    while t < t_stop:
        
        # Démarrer la trace après quelques itérations de warmup
        if n_iter == 5 and not trace_done:
            jax.profiler.start_trace("/tmp/while-trace")

        with jax.profiler.TraceAnnotation("update"):
            Mass, Momx, Momy, Momz, Energy, dt, Bx, By, Bz = update(
                Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac, Bx, By, Bz
            )
        t += dt
        n_iter += 1
        jax.block_until_ready((Mass, Momx, Momy, Momz, Energy, Bx, By, Bz))

        # Arrêter après 10 itérations tracées
        if n_iter == 15 and not trace_done:
            jax.profiler.stop_trace()
            trace_done = True

    global_end = time.time()

    # KPIs
    total_time = global_end - global_start
    mcups = (Nx * Ny * Nz * n_iter) / (1e6 * total_time)
    nvar = 8
    sizeof_double = 8
    SoL = (2 * Nx * Ny * Nz * nvar * sizeof_double) / (total_time * 1e9)

    print(f"\nSimulation complete after {n_iter} iterations")
    print(f"Final time reached: {t:.4f}")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Performance: {mcups:.2f} million cell updates per second (MCUPS)")
    print(f"Performance: {SoL:.2f} GB/s (SoL)")


if __name__ == "__main__":
    args, config = load_config_and_args()
    main(args, config)
    # ms = jax.devices("gpu")[0].memory_stats()
    # print(f"\n[GPU memory] in use = {ms['bytes_in_use']/1e9:.2f} GB | peak = {ms['peak_bytes_in_use']/1e9:.2f} GB")