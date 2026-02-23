#!/usr/bin/env python3
import os
import time
import json
from functools import partial
from pathlib import Path, PurePath

# ------------------------------------------------------------------
# PROFILING FLAG (lu AVANT toute logique)
# ------------------------------------------------------------------
PROFILE = os.environ.get("PROFILE_STEP", "0") == "1"

# ------------------------------------------------------------------
# IMPORTANT : définir XLA_FLAGS AVANT l'initialisation JAX si possible
# (sinon, passe-les depuis le shell, recommandé pour Nsight)
# ------------------------------------------------------------------
flags = os.environ.get("XLA_FLAGS", "")
flags += (
    "--xla_gpu_triton_gemm_any=false "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)
os.environ["XLA_FLAGS"] = flags

import jax
import jax.numpy as jnp

from modules import *
from numerical_scheme import *
from initial_conditions import *
from repartition_gpu import *
from load_config import *
from physics import *

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding


# ==================================================================
# MAIN
# ==================================================================
def main(args, config):

    # -----------------------------
    # CPU / GPU selection
    # -----------------------------
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # -----------------------------
    # Simulation parameters
    # -----------------------------
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

    # -----------------------------
    # Domain
    # -----------------------------
    xlin = jnp.linspace(0.5 * dx, boxsize - 0.5 * dx, Nx)
    ylin = jnp.linspace(0.5 * dy, boxsize - 0.5 * dy, Ny)
    zlin = jnp.linspace(0.5 * dz, boxsize - 0.5 * dz, Nz)
    X, Y, Z = jnp.meshgrid(xlin, ylin, zlin, indexing="ij")

    # -----------------------------
    # Sharding (safe si 1 GPU)
    # -----------------------------
    n_devices = jax.device_count()
    x_opt, y_opt, z_opt = optimal_3d_partition(n_devices)

    mesh = Mesh(
        mesh_utils.create_device_mesh((x_opt, y_opt, z_opt)),
        ("x", "y", "z"),
    )
    sharding = NamedSharding(mesh, PartitionSpec("x", "y", "z"))

    X = jax.lax.with_sharding_constraint(X, sharding)
    Y = jax.lax.with_sharding_constraint(Y, sharding)
    Z = jax.lax.with_sharding_constraint(Z, sharding)

    # -----------------------------
    # Initial conditions
    # -----------------------------
    rho, vx, vy, vz, Bx, By, Bz, P = inital_condition(
        IC, X, Y, Z, gamma, boxsize
    )
    Mass, Momx, Momy, Momz, Energy, Bx, By, Bz = get_conserved(
        rho, vx, vy, vz, P, gamma, Bx, By, Bz
    )

    # ==============================================================
    # PROFILING MODE (Nsight Compute / Roofline)
    # ==============================================================
    if PROFILE:
        print("[PROFILE] Running warmup + 1 timestep for Nsight Compute")

        # -------- warmup (compile) --------
        Mass, Momx, Momy, Momz, Energy, dt, Bx, By, Bz = update(
            Mass, Momx, Momy, Momz, Energy,
            dx, dy, dz, gamma, courant_fac,
            Bx, By, Bz
        )
        jax.block_until_ready(
            (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz)
        )

        # -------- one clean timestep --------
        Mass, Momx, Momy, Momz, Energy, dt, Bx, By, Bz = update(
            Mass, Momx, Momy, Momz, Energy,
            dx, dy, dz, gamma, courant_fac,
            Bx, By, Bz
        )
        jax.block_until_ready(
            (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz)
        )

        print("[PROFILE] Done. Exiting before time loop.")
        return

    # ==============================================================
    # NORMAL MODE (full simulation)
    # ==============================================================
    global_start = time.time()
    t = 0.0
    n_iter = 0

    while t < t_stop:
        Mass, Momx, Momy, Momz, Energy, dt, Bx, By, Bz = update(
            Mass, Momx, Momy, Momz, Energy,
            dx, dy, dz, gamma, courant_fac,
            Bx, By, Bz
        )
        t += dt
        n_iter += 1

    jax.block_until_ready((Mass, Momx, Momy, Momz, Energy, Bx, By, Bz))

    global_end = time.time()
    total_time = global_end - global_start

    # -----------------------------
    # KPIs
    # -----------------------------
    mcups = (Nx * Ny * Nz * n_iter) / (1e6 * total_time)

    nvar = 8
    sizeof_double = 8 if use_double else 4
    SoL = (2 * Nx * Ny * Nz * nvar * sizeof_double) / (total_time * 1e9)

    print(f"\nSimulation complete after {n_iter} iterations")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Performance: {mcups:.2f} MCUPS")
    print(f"Lower-bound streaming BW: {SoL:.2f} GB/s")


# ==================================================================
# ENTRY POINT
# ==================================================================
if __name__ == "__main__":
    args, config = load_config_and_args()
    main(args, config)

    # Optional: GPU memory stats (avoid in profiling mode)
    if not PROFILE and jax.devices("gpu"):
        ms = jax.devices("gpu")[0].memory_stats()
        print(
            f"\n[GPU memory] in use = {ms['bytes_in_use']/1e9:.2f} GB | "
            f"peak = {ms['peak_bytes_in_use']/1e9:.2f} GB"
        )