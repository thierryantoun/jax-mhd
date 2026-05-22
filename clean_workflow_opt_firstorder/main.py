#!/usr/bin/env python3

import os
import time
import jax
from functools import partial

from modules import *
from numerical_scheme import *
from initial_conditions import *
from repartition_gpu import *
from load_config import *
from physics import *

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

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
    rho, vx, vy, vz, P = inital_condition(IC, X, Y, Z, gamma, boxsize)
    Mass, Momx, Momy, Momz, Energy = get_conserved(rho, vx, vy, vz, P, gamma)

    # Initial state : (conserved vars..., t, count)
    initial_state = (Mass, Momx, Momy, Momz, Energy, jnp.array(0.0), jnp.array(0))

    # Estimate max_steps conservatively
    c02 = gamma * P / rho
    cmf = jnp.sqrt(c02)
    val_max = jnp.maximum(
        jnp.maximum(cmf + jnp.abs(vx), cmf + jnp.abs(vy)),
        cmf + jnp.abs(vz)
    )
    dt_est = courant_fac * jnp.min(jnp.array([dx, dy, dz])) / jnp.max(val_max)
    max_steps = int(jnp.ceil(t_stop / dt_est)) + 5

    print(f"Grid: {Nx}x{Ny}x{Nz}  |  max_steps estimated: {max_steps}")

    @partial(jax.jit, static_argnames=["dx", "dy", "dz", "gamma", "courant_fac"])
    def scan_step(state, _, dx, dy, dz, gamma, courant_fac):
        Mass, Momx, Momy, Momz, Energy, t, count = state
        Mass, Momx, Momy, Momz, Energy, dt, rho = update(
            Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac
        )
        t += dt
        count += 1
        return (Mass, Momx, Momy, Momz, Energy, t, count), None

    @partial(jax.jit, static_argnames=["dx", "dy", "dz", "gamma", "courant_fac", "max_steps"])
    def run_simulation(initial_state, dx, dy, dz, gamma, courant_fac, max_steps):
        return jax.lax.scan(
            lambda s, _: scan_step(s, _, dx, dy, dz, gamma, courant_fac),
            initial_state,
            None,
            length=max_steps
        )

    # -------------------------
    # WARMUP  (compilation seule)
    # -------------------------
    t0 = time.perf_counter()
    lowered = jax.jit(
        run_simulation,
        static_argnames=["dx", "dy", "dz", "gamma", "courant_fac", "max_steps"]
    ).lower(initial_state, dx, dy, dz, gamma, courant_fac, max_steps)
    compiled = lowered.compile()
    compile_time = time.perf_counter() - t0

    # Un premier run pour s'assurer que tout est chaud (caches GPU, etc.)
    state, _ = compiled(initial_state)
    jax.block_until_ready(state)

    # -------------------------
    # RUN CHRONOMÉTRÉ
    # -------------------------
    global_start = time.perf_counter()

    state, _ = run_simulation(
        initial_state, dx, dy, dz, gamma, courant_fac, max_steps
    )
    jax.block_until_ready(state)

    global_end = time.perf_counter()

    # KPIs
    t_final    = state[5]
    n_iter     = int(state[6])
    total_time = global_end - global_start
    mcups      = (Nx * Ny * Nz * n_iter) / (1e6 * total_time)

    print("\nSimulation complete")
    print(f"Final time reached : {float(t_final):.4f}")
    print(f"nb_iterations      : {n_iter}")
    print(f"Total runtime      : {total_time:.4f} s")
    print(f"Performance        : {mcups:.2f} MCUPS")
    print(f"Compilation time   : {compile_time:.2f} s")


if __name__ == "__main__":
    args, config = load_config_and_args()
    main(args, config)
