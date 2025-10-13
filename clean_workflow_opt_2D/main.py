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
from matplotlib.pyplot import *

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

    # Domain
    xlin = jnp.linspace(0.5 * dx, boxsize - 0.5 * dx, Nx)
    ylin = jnp.linspace(0.5 * dy, boxsize - 0.5 * dy, Ny)
    X, Y = jnp.meshgrid(xlin, ylin, indexing="ij")

    n_devices = jax.device_count()
    x_opt, y_opt = optimal_3d_partition(n_devices)
    mesh = Mesh(mesh_utils.create_device_mesh((x_opt, y_opt)), ("x", "y"))
    sharding = NamedSharding(mesh, PartitionSpec("x", "y"))

    X = jax.lax.with_sharding_constraint(X, sharding)
    Y = jax.lax.with_sharding_constraint(Y, sharding)

    # Initial conditions
    rho, vx, vy, Bx, By, P = inital_condition(IC, X, Y, gamma, boxsize)
    Mass, Momx, Momy, Energy, Bx, By = get_conserved(rho, vx, vy, P, gamma, Bx, By)

    # Initial state
    initial_state = (Mass, Momx, Momy, Energy, Bx, By, jnp.array(0.0), jnp.array(0))

    # Estimate max_steps conservatively
    c02 = gamma * P / rho
    ca2 = (Bx**2 + By**2) / rho
    cap2x = Bx**2 / rho
    cap2y = By**2 / rho

    cmfx = jnp.sqrt(0.5*(c02 + ca2) + 0.5*jnp.sqrt((c02 + ca2)**2 - 4*c02*cap2x))
    cmfy = jnp.sqrt(0.5*(c02 + ca2) + 0.5*jnp.sqrt((c02 + ca2)**2 - 4*c02*cap2y))

    val_max = jnp.maximum(cmfx + jnp.abs(vx), cmfy + jnp.abs(vy))

    dt_est = courant_fac * jnp.min(jnp.array([dx, dy])) / jnp.max(val_max)

    # max_steps = int(jnp.ceil(t_stop / dt_est)) + 5
    max_steps = 10

    @partial(jax.jit, static_argnames=["dx", "dy", "gamma", "courant_fac"])
    def scan_step(state, _, dx, dy, gamma, courant_fac):
        Mass, Momx, Momy, Energy, Bx, By, t, count = state
        Mass, Momx, Momy, Energy, dt, rho, Bx, By = update(
            Mass, Momx, Momy, Energy, dx, dy, gamma, courant_fac, Bx, By
        )
        t += dt
        count += 1
        return (Mass, Momx, Momy, Energy, Bx, By, t, count), None

    global_start = time.time()
    final_state, _ = jax.lax.scan(
        lambda s, _: scan_step(s, _, dx, dy, gamma, courant_fac),
        initial_state, None, length=max_steps
    )
    jax.block_until_ready(final_state)
    global_end = time.time()

    # KPIs temporels
    t_final = final_state[6]
    n_iter  = int(final_state[7])
    total_time = global_end - global_start
    mcups = (Nx * Ny * n_iter) / (1e6 * total_time)
    
    print("\nSimulation complete")
    print(f"Final time reached: {float(t_final):.4f}")
    print("nb_iterations:", n_iter)
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Performance: {mcups:.2f} million cell updates per second (MCUPS)")
    
    figure(1)
    clf()
    imshow(final_state[0],origin='lower')
    colorbar()
    savefig('output_'+'.png')

if __name__ == "__main__":
    args, config = load_config_and_args()
    main(args, config)