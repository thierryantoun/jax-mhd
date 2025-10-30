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
    rho, vx, vy, vz, Bx, By, Bz, P = inital_condition(IC, X, Y, Z, gamma, boxsize)
    Mass, Momx, Momy, Momz, Energy, Bx, By, Bz = get_conserved(rho, vx, vy, vz, P, gamma, Bx, By, Bz)

    # Initial state
    initial_state = (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, jnp.array(0.0), jnp.array(0))

    # Estimate max_steps conservatively
    c02 = gamma * P / rho
    ca2 = (Bx**2 + By**2 + Bz**2) / rho
    cap2x = Bx**2 / rho
    cap2y = By**2 / rho
    cap2z = Bz**2 / rho

    cmfx = jnp.sqrt(0.5*(c02 + ca2) + 0.5*jnp.sqrt((c02 + ca2)**2 - 4*c02*cap2x))
    cmfy = jnp.sqrt(0.5*(c02 + ca2) + 0.5*jnp.sqrt((c02 + ca2)**2 - 4*c02*cap2y))
    cmfz = jnp.sqrt(0.5*(c02 + ca2) + 0.5*jnp.sqrt((c02 + ca2)**2 - 4*c02*cap2z))

    val_max = jnp.maximum(
        jnp.maximum(cmfx + jnp.abs(vx), cmfy + jnp.abs(vy)),
        cmfz + jnp.abs(vz)
    )

    dt_est = courant_fac * jnp.min(jnp.array([dx, dy, dz])) / jnp.max(val_max)

    max_steps = int(jnp.ceil(t_stop / dt_est)) + 5
    #max_steps = 1

    @partial(jax.jit, static_argnames=["dx", "dy", "dz", "gamma", "courant_fac"])
    def scan_step(state, _, dx, dy, dz, gamma, courant_fac):
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count = state
        Mass, Momx, Momy, Momz, Energy, dt, rho, Bx, By, Bz = update(
            Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac, Bx, By, Bz
        )
        t += dt
        count += 1
        return (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz, t, count), None
    
    
    t0 = time.perf_counter()
    compiled = scan_step.lower(
        initial_state, None, dx=dx, dy=dy, dz=dz,
        gamma=gamma, courant_fac=courant_fac).compile()
    t1 = time.perf_counter()
    print(f"[JAX] Compile time (step) = {t1 - t0:.3f} s")
    
    # il faut que max_steps soit static pour le lax.scan
    def run_scan(state, dx, dy, dz, gamma, courant_fac, max_steps: int):
        return jax.lax.scan(lambda s, _: scan_step(s, _, dx, dy, dz, gamma, courant_fac),
                            state, None, length=max_steps)

    # on marque les hyperparams comme statiques pour le traçage
    run_scan_jit = jax.jit(run_scan,
        static_argnames=("dx","dy","dz","gamma","courant_fac","max_steps"))

    # ---------- compilation de la boucle scan ----------
    t0 = time.perf_counter()
    compiled = run_scan_jit.lower(
        initial_state, dx=dx, dy=dy, dz=dz,
        gamma=gamma, courant_fac=courant_fac, max_steps=int(max_steps)
    ).compile()
    t1 = time.perf_counter()
    print(f"[JAX] Compile time (scan) = {t1 - t0:.3f} s")

    global_start = time.time()
    final_state, _ = jax.lax.scan(
        lambda s, _: scan_step(s, _, dx, dy, dz, gamma, courant_fac),
        initial_state, None, length=max_steps
    )
    jax.block_until_ready(final_state)
    # jax.profiler.save_device_memory_profile("mem_after_run.prof")
    # print("[mem] wrote mem_after_run.prof")
    global_end = time.time()

    # KPIs temporels
    t_final = final_state[8]
    n_iter  = int(final_state[9])
    total_time = global_end - global_start
    mcups = (Nx * Ny * Nz * n_iter) / (1e6 * total_time)
    
    print("\nSimulation complete")
    print(f"Final time reached: {float(t_final):.4f}")
    print("nb_iterations:", n_iter)
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Performance: {mcups:.2f} million cell updates per second (MCUPS)")
    
    if args.benchmark == False:
    
        import pyvista as pv
        import numpy as np

        # Convertir les arrays JAX en NumPy
        Bx_np = np.array(final_state[5])
        By_np = np.array(final_state[6])
        Bz_np = np.array(final_state[7])
        rho_np = np.array(final_state[0])  # Mass / volume = rho (si unit vol)

        # Calculer la densité si nécessaire
        rho_np = rho_np  # ou Mass / dx**3 si tu veux une vraie densité

        # Créer la grille PyVista
        grid = pv.ImageData()

        grid.dimensions = np.array(rho_np.shape) + 1 
        grid.origin = (0, 0, 0)
        grid.spacing = (dx, dx, dx)

        # Taille et dimensions
        grid.dimensions = np.array(Bx_np.shape) + 1  # +1 car c’est des cellules
        grid.spacing = (dx, dy, dz)
        grid.origin = (0, 0, 0)

        # Ajouter les champs
        grid["Bx"] = Bx_np.ravel(order="F")
        grid["By"] = By_np.ravel(order="F")
        grid["Bz"] = Bz_np.ravel(order="F")
        grid["rho"] = rho_np.ravel(order="F")

        # Export en fichier .vti
        grid.save("output_final.vti")

if __name__ == "__main__":
    args, config = load_config_and_args()
    main(args, config)
    # ms = jax.devices("gpu")[0].memory_stats()
    # print(f"\n[GPU memory] in use = {ms['bytes_in_use']/1e9:.2f} GB | peak = {ms['peak_bytes_in_use']/1e9:.2f} GB")