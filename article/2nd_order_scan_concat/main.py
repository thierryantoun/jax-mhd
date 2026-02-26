# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import jax
import jax.numpy as jnp
from functools import partial

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

from load_config import load_config_and_args
from repartition_gpu import optimal_3d_partition
from initial_conditions import inital_condition
from physics import get_conserved

from numerical_scheme import update_jit

# ----------------------------
# Profiling helper (host NVTX)
# ----------------------------
def run_profile_loop(initial_state, dx, dy, dz, gamma, courant_fac, n_warmup=5, n_profile=3):
    import nvtx  # import here so normal runs don't require nvtx installed

    state = initial_state

    # Warmup: compile + stabilize (OUT of NVTX)
    for _ in range(n_warmup):
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count = state
        Mass, Momx, Momy, Momz, Energy, dt, Bx, By, Bz = update_jit(
            Mass, Momx, Momy, Momz, Energy,
            dx=dx, dy=dy, dz=dz, gamma=gamma, courant_fac=courant_fac,
            Bx=Bx, By=By, Bz=Bz
        )
        t = t + dt
        count = count + 1
        state = (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count)
        jax.block_until_ready(state)  # force GPU completion outside measured region

    # Profile: NVTX around steady-state iterations
    for i in range(n_profile):
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count = state
        with nvtx.annotate(f"update_step_{i}", color="red"):
            Mass, Momx, Momy, Momz, Energy, dt, Bx, By, Bz = update_jit(
                Mass, Momx, Momy, Momz, Energy,
                dx=dx, dy=dy, dz=dz, gamma=gamma, courant_fac=courant_fac,
                Bx=Bx, By=By, Bz=Bz
            )
            t = t + dt
            count = count + 1
            state = (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count)
            jax.block_until_ready(state)  # ensure kernels land inside NVTX
    return state


def main(args, config):
    USE_CPU_ONLY = args.cpu

    flags = os.environ.get("XLA_FLAGS", "")
    if USE_CPU_ONLY:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        flags += (
            "--xla_gpu_triton_gemm_any=false "
            "--xla_gpu_enable_latency_hiding_scheduler=true "
            "--xla_gpu_enable_highest_priority_async_stream=true "
        )
    os.environ["XLA_FLAGS"] = flags

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
        sizeof_float = 8
    else:
        print("Using single precision")
        sizeof_float = 4

    dx = boxsize / Nx
    dy = boxsize / Ny
    dz = boxsize / Nz

    # Domain
    xlin = jnp.linspace(0.5 * dx, boxsize - 0.5 * dx, Nx)
    ylin = jnp.linspace(0.5 * dy, boxsize - 0.5 * dy, Ny)
    zlin = jnp.linspace(0.5 * dz, boxsize - 0.5 * dz, Nz)
    X, Y, Z = jnp.meshgrid(xlin, ylin, zlin, indexing="ij")

    # Sharding (as you had)
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

    # Initial state
    initial_state = (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, jnp.array(0.0), jnp.array(0))

    # Estimate max_steps conservatively (same as yours)
    c02 = gamma * P / rho
    ca2 = (Bx**2 + By**2 + Bz**2) / rho
    cap2x = Bx**2 / rho
    cap2y = By**2 / rho
    cap2z = Bz**2 / rho

    cmfx = jnp.sqrt(0.5*(c02 + ca2) + 0.5*jnp.sqrt((c02 + ca2)**2 - 4*c02*cap2x))
    cmfy = jnp.sqrt(0.5*(c02 + ca2) + 0.5*jnp.sqrt((c02 + ca2)**2 - 4*c02*cap2y))
    cmfz = jnp.sqrt(0.5*(c02 + ca2) + 0.5*jnp.sqrt((c02 + ca2)**2 - 4*c02*cap2z))

    val_max = jnp.maximum(jnp.maximum(cmfx + jnp.abs(vx), cmfy + jnp.abs(vy)), cmfz + jnp.abs(vz))
    dt_est = courant_fac * jnp.min(jnp.array([dx, dy, dz])) / jnp.max(val_max)
    max_steps = int(jnp.ceil(t_stop / dt_est)) + 5

    # ----------------------------
    # Two modes:
    #  - normal: JIT scan (best end-to-end perf)
    #  - profile: host loop + NVTX (best attribution)
    # ----------------------------
    if getattr(args, "profile", False):
        print("\n[PROFILE MODE] Host loop + NVTX regions (use nsys --trace=cuda,nvtx)")
        global_start = time.time()
        final_state = run_profile_loop(
            initial_state, dx, dy, dz, gamma, courant_fac,
            n_warmup=getattr(args, "warmup", 5),
            n_profile=getattr(args, "profile_steps", 3),
        )
        jax.block_until_ready(final_state)
        global_end = time.time()

        # KPIs on the profiled loop count
        t_final = float(final_state[8])
        n_iter = int(final_state[9])
        total_time = global_end - global_start
        mcups = (Nx * Ny * Nz * n_iter) / (1e6 * total_time)
        nvar = 8
        SoL = (2 * Nx * Ny * Nz * nvar * sizeof_float) / (total_time * 1e9)

        print("\nProfile run complete")
        print(f"Final time reached: {t_final:.4f}")
        print("nb_iterations:", n_iter)
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"Performance: {mcups:.2f} MCUPS")
        print(f"Performance: {SoL:.2f} GB/s (SoL)")
        return

    # Normal mode: scan (fast)
    @partial(jax.jit, static_argnames=("dx", "dy", "dz", "gamma", "courant_fac"))
    def scan_step(state, _, dx, dy, dz, gamma, courant_fac):
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count = state
        Mass, Momx, Momy, Momz, Energy, dt, Bx, By, Bz = update_jit(
            Mass, Momx, Momy, Momz, Energy,
            dx=dx, dy=dy, dz=dz, gamma=gamma, courant_fac=courant_fac,
            Bx=Bx, By=By, Bz=Bz
        )
        t = t + dt
        count = count + 1
        return (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count), None

    global_start = time.time()
    final_state, _ = jax.lax.scan(
        lambda s, _: scan_step(s, _, dx, dy, dz, gamma, courant_fac),
        initial_state,
        xs=None,
        length=max_steps
    )
    jax.block_until_ready(final_state)
    global_end = time.time()

    # KPIs
    t_final = float(final_state[8])
    n_iter = int(final_state[9])
    total_time = global_end - global_start
    mcups = (Nx * Ny * Nz * n_iter) / (1e6 * total_time)
    nvar = 8
    SoL = (2 * Nx * Ny * Nz * nvar * sizeof_float) / (total_time * 1e9)

    print("\nSimulation complete")
    print(f"Final time reached: {t_final:.4f}")
    print("nb_iterations:", n_iter)
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Performance: {mcups:.2f} MCUPS")
    print(f"Performance: {SoL:.2f} GB/s (SoL)")


if __name__ == "__main__":
    args, config = load_config_and_args()

    # If your current arg parser doesn't include these, add them there:
    # --profile           : enable profile mode
    # --warmup N          : warmup iterations (default 5)
    # --profile-steps N   : profiled iterations (default 3)
    #
    # This code assumes args.profile / args.warmup / args.profile_steps may exist.
    main(args, config)