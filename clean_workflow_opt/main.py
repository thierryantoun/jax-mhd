#!/usr/bin/env python

import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from modules import *
from numerical_scheme import update
from initial_conditions import *
from repartition_gpu import *
from load_config import *
from physics import get_conserved, get_primitive
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

def main(args, config):
    USE_CPU_ONLY = args.cpu

    flags = os.environ.get("XLA_FLAGS", "")
    if USE_CPU_ONLY:
        flags += " --xla_force_host_platform_device_count=8"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        flags += (
            "--xla_gpu_triton_gemm_any=false "
            "--xla_gpu_enable_latency_hiding_scheduler=true "
            "--xla_gpu_enable_highest_priority_async_stream=true "
        )
    os.environ["XLA_FLAGS"] = flags

    IC = config["simulation"]["IC"]
    N = int(config["simulation"]["resolution"])
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

    dx = boxsize / N

    # Domain
    xlin = jnp.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    X, Y, Z = jnp.meshgrid(xlin, xlin, xlin, indexing="ij")

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

    # Estimate max_steps conservatively
    cmax = jnp.sqrt(gamma * jnp.max(P) / jnp.min(rho)) + jnp.max(jnp.abs(vx))
    dt_est = courant_fac * dx / cmax
    max_steps = int(jnp.ceil(t_stop / dt_est)) + 30

    @partial(jax.jit, static_argnames=["dx", "gamma", "courant_fac"])
    def scan_step(state, _, dx, gamma, courant_fac):
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count = state
        Mass, Momx, Momy, Momz, Energy, dt, rho, Bx, By, Bz = update(
            Mass, Momx, Momy, Momz, Energy, dx, gamma, courant_fac, Bx, By, Bz
        )
        t += dt
        count += 1
        return (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count), None

    global_start = time.time()
    final_state, _ = jax.lax.scan(
        lambda s, _: scan_step(s, _, dx, gamma, courant_fac),
        initial_state,
        None,
        length=max_steps
    )
    global_end = time.time()

    _, _, _, _, _, _, _, _, t_final, n_iter = final_state
    total_time = global_end - global_start
    mcups = (N**3 * int(n_iter)) / (1e6 * total_time)

    print("\nSimulation complete")
    print(f"Final time reached: {float(t_final):.4f}")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Performance: {mcups:.2f} million cell updates per second (MCUPS)")

if __name__ == "__main__":
    args, config = load_config_and_args()
    main(args, config)