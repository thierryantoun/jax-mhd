#!/usr/bin/env python3

import os
import time
import jax
import jax.numpy as jnp
from functools import partial
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from modules import *
from numerical_scheme import update
from initial_conditions import inital_condition
from repartition_gpu import optimal_3d_partition
from load_config import load_config_and_args
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

    IC          = config["simulation"]["IC"]
    Nx          = int(config["simulation"]["resolution_x"])
    Ny          = int(config["simulation"]["resolution_y"])
    use_double  = config.getboolean("simulation", "double")
    boxsize     = float(config["simulation"]["boxsize"])
    gamma       = float(config["simulation"]["gamma"])
    courant_fac = float(config["simulation"]["courant_fac"])
    t_stop      = float(config["simulation"]["t_stop"])
    
    tau  = float(config["simulation"].get("tau", float("inf")))  # désactive le chauffage si non fourni
    grav = float(config["simulation"].get("grav", "-1.0"))
    cv   = float(config["simulation"].get("cv", "1.5"))

    if use_double:
        print("Using double precision")
        jax.config.update("jax_enable_x64", True)
    else:
        print("Using single precision")

    dx = (boxsize + 1) / Nx
    dy = boxsize / Ny

    # Domaine
    xlin = jnp.linspace(0.5 * dx, boxsize+1 - 0.5 * dx, Nx)
    ylin = jnp.linspace(0.5 * dy, boxsize - 0.5 * dy, Ny)
    X, Y = jnp.meshgrid(xlin, ylin, indexing="ij")

    n_devices = jax.device_count()
    x_opt, y_opt = optimal_3d_partition(n_devices)
    mesh = Mesh(mesh_utils.create_device_mesh((x_opt, y_opt)), ("x", "y"))
    sharding = NamedSharding(mesh, PartitionSpec("x", "y"))

    X = jax.lax.with_sharding_constraint(X, sharding)
    Y = jax.lax.with_sharding_constraint(Y, sharding)

    rho, vx, vy, P, Teq, phi, Energy = inital_condition(IC, X, Y, gamma, boxsize)

    Mass = rho
    Momx = rho * vx
    Momy = rho * vy

    initial_state = (Mass, Momx, Momy, Energy, jnp.array(0.0), jnp.array(0))

    max_steps = 100

    for i in range(max_steps):
        # M = np.asarray(jax.device_get(Mass))
        # U = np.asarray(jax.device_get(Momx))
        # V = np.asarray(jax.device_get(Momy))
        # E = np.asarray(jax.device_get(Energy))
        # PH = np.asarray(jax.device_get(phi))

        # T = (E - 0.5 * (U**2 + V**2) / M - PH * M) / (cv * M)
        # plt.figure(figsize=(7, 5))
        # plt.clf()
        # im = plt.imshow(T.T, origin="lower")
        # nx, ny = U.shape
        # X = np.arange(nx)
        # Y = np.arange(ny)
        # plt.streamplot(X, Y, U.T, V.T, density=1.2, linewidth=0.7)
        # plt.colorbar(im)
        # plt.tight_layout()
        # fname = f"output_{i:06d}.png"
        # plt.savefig(fname, dpi=150)
        # plt.close()

        Mass, Momx, Momy, Energy, dt, rho = update(
            Mass, Momx, Momy, Energy, dx, dy, gamma, courant_fac, phi, tau, Teq, grav=-1.0, cv=1.5
        )

    print("\nSimulation complete")

    save_animation_path = "output_euler"

    if not os.path.exists(save_animation_path):
        os.makedirs(save_animation_path, exist_ok=True)

    Path(save_animation_path).mkdir(parents=True, exist_ok=True)
    img = jnp.rot90(rho)
    plt.imsave(
        str(Path(save_animation_path) / "rho_final.png"),
        np.asarray(jax.device_get(img)),
        cmap="jet",
        vmin=0.8,
        vmax=2.2,
    )
    print(f"Saved: {Path(save_animation_path) / 'rho_final.png'}")

if __name__ == "__main__":
    args, config = load_config_and_args()
    main(args, config)
