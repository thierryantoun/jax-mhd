#!/usr/bin/env python3
import os
import time
import json
from functools import partial
from pathlib import Path, PurePath

# ==============================================================
# PROFILING MODE FLAG
# ==============================================================
PROFILE = os.environ.get("PROFILE_STEP", "0") == "1"

# ==============================================================
# (Optionnel) XLA_FLAGS
# NOTE: idéalement passé depuis le shell, mais on garde ton pattern
# ==============================================================
flags = os.environ.get("XLA_FLAGS", "")
flags += (
    "--xla_gpu_triton_gemm_any=false "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)
os.environ["XLA_FLAGS"] = flags

# --------------------------------------------------------------
# En mode profiling: forcer 1 GPU visible (mono-GPU strict)
# (si ton job a >1 GPU alloué, ça évite interconnect/ctc__*)
# --------------------------------------------------------------
if PROFILE and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp

from modules import *
from numerical_scheme import *          # contient update()
from initial_conditions import *
from repartition_gpu import *
from load_config import *
from physics import *

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding


def _block_until_ready(tree):
    """Synchronise proprement n'importe quel pytree JAX."""
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree
    )


def main(args, config):
    USE_CPU_ONLY = args.cpu

    # -----------------------------
    # CPU / GPU selection
    # -----------------------------
    if USE_CPU_ONLY:
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

    # ==============================================================
    # SHARDING
    # - NORMAL: on garde ton mesh/sharding
    # - PROFILE: on désactive complètement sharding/mesh
    # ==============================================================
    if not PROFILE:
        n_devices = jax.device_count()
        x_opt, y_opt, z_opt = optimal_3d_partition(n_devices)
        mesh = Mesh(mesh_utils.create_device_mesh((x_opt, y_opt, z_opt)), ("x", "y", "z"))
        sharding = NamedSharding(mesh, PartitionSpec("x", "y", "z"))

        X = jax.lax.with_sharding_constraint(X, sharding)
        Y = jax.lax.with_sharding_constraint(Y, sharding)
        Z = jax.lax.with_sharding_constraint(Z, sharding)
    else:
        print("[PROFILE] Sharding disabled (mono-GPU local arrays).")
        print("[PROFILE] devices:", jax.devices())
        print("[PROFILE] device_count:", jax.device_count())

    # -----------------------------
    # Initial conditions
    # -----------------------------
    rho, vx, vy, vz, Bx, By, Bz, P = inital_condition(IC, X, Y, Z, gamma, boxsize)
    Mass, Momx, Momy, Momz, Energy, Bx, By, Bz = get_conserved(rho, vx, vy, vz, P, gamma, Bx, By, Bz)

    # ==============================================================
    # PROFILING MODE: warmup + 1 step, puis exit
    # ==============================================================
    if PROFILE:
        print("[PROFILE] Warmup (compile) + 1 timestep, then exit.")

        # warmup compile (à ignorer dans Nsight via --launch-skip 1)
        out = update(Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac, Bx, By, Bz)
        _block_until_ready(out)

        # 1 timestep "clean"
        out = update(Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac, Bx, By, Bz)
        _block_until_ready(out)

        print("[PROFILE] Done. Exiting before scan.")
        return

    # ==============================================================
    # NORMAL MODE: lax.scan (ton code)
    # ==============================================================
    initial_state = (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, jnp.array(0.0), jnp.array(0))

    # Estimate max_steps conservatively
    c02 = gamma * P / rho
    ca2 = (Bx**2 + By**2 + Bz**2) / rho
    cap2x = Bx**2 / rho
    cap2y = By**2 / rho
    cap2z = Bz**2 / rho

    cmfx = jnp.sqrt(0.5 * (c02 + ca2) + 0.5 * jnp.sqrt((c02 + ca2) ** 2 - 4 * c02 * cap2x))
    cmfy = jnp.sqrt(0.5 * (c02 + ca2) + 0.5 * jnp.sqrt((c02 + ca2) ** 2 - 4 * c02 * cap2y))
    cmfz = jnp.sqrt(0.5 * (c02 + ca2) + 0.5 * jnp.sqrt((c02 + ca2) ** 2 - 4 * c02 * cap2z))

    val_max = jnp.maximum(
        jnp.maximum(cmfx + jnp.abs(vx), cmfy + jnp.abs(vy)),
        cmfz + jnp.abs(vz),
    )

    dt_est = courant_fac * jnp.min(jnp.array([dx, dy, dz])) / jnp.max(val_max)
    max_steps = int(jnp.ceil(t_stop / dt_est)) + 5

    @partial(jax.jit, static_argnames=["dx", "dy", "dz", "gamma", "courant_fac"])
    def scan_step(state, _, dx, dy, dz, gamma, courant_fac):
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count = state
        Mass, Momx, Momy, Momz, Energy, dt, Bx, By, Bz = update(
            Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac, Bx, By, Bz
        )
        t = t + dt
        count = count + 1
        return (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count), None

    global_start = time.time()
    final_state, _ = jax.lax.scan(
        lambda s, _: scan_step(s, _, dx, dy, dz, gamma, courant_fac),
        initial_state,
        None,
        length=max_steps,
    )
    jax.block_until_ready(final_state)
    global_end = time.time()

    # KPIs temporels
    t_final = final_state[8]
    n_iter = int(final_state[9])
    total_time = global_end - global_start

    mcups = (Nx * Ny * Nz * n_iter) / (1e6 * total_time)
    nvar = 8
    sizeof_double = 8 if use_double else 4
    SoL = (2 * Nx * Ny * Nz * nvar * sizeof_double) / (total_time * 1e9)

    print("\nSimulation complete")
    print(f"Final time reached: {float(t_final):.4f}")
    print("nb_iterations:", n_iter)
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Performance: {mcups:.2f} million cell updates per second (MCUPS)")
    print(f"Performance: {SoL:.2f} GB/s (SoL)")


if __name__ == "__main__":
    args, config = load_config_and_args()
    main(args, config)

    # Evite de polluer le profiling Nsight avec des calls driver en fin de run
    if not PROFILE and jax.devices("gpu"):
        ms = jax.devices("gpu")[0].memory_stats()
        print(
            f"\n[GPU memory] in use = {ms['bytes_in_use']/1e9:.2f} GB | "
            f"peak = {ms['peak_bytes_in_use']/1e9:.2f} GB"
        )