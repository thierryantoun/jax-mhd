#!/usr/bin/env python
# A simple example of solving the Euler equations with JAX
# Philip Mocz (2024)

from modules import *
from numerical_scheme import *
from initial_conditions import *  
from repartition_gpu import *
from load_config import *
import os
import time
import numpy as np
import jax

from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding

def main(args, config):
    """Finite Volume simulation"""
    
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

    if args.benchmark:
        print("Benchmark mode enabled: disabling VTK output.")
        save_freq = 0.0
    else:
        save_freq = float(config["simulation"]["save_freq"])
    
    n_devices = jax.device_count()
    x_opt, y_opt, z_opt = optimal_3d_partition(n_devices)
    mesh = Mesh(mesh_utils.create_device_mesh((x_opt, y_opt, z_opt)), ("x", "y", "z"))
    sharding = NamedSharding(mesh, PartitionSpec("x", "y", "z"))

    if jax.process_index() == 0:
        for env_var in [
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_NODELIST",
            "SLURM_STEP_NODELIST",
            "SLURM_STEP_GPUS",
            "SLURM_GPUS",
        ]:
            print(f'{env_var}: {os.getenv(env_var,"")}')
        print("Total number of processes: ", jax.process_count())
        print("Total number of devices: ", jax.device_count())
        print("List of devices: ", jax.devices())
        print("Number of devices on this process: ", jax.local_device_count())

    save_animation_path = (
        "output_euler_" + str(N) + ("double" if use_double else "single")
    )

    # Mesh
    dx = boxsize / N
    xlin = jnp.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    X, Y, Z = jnp.meshgrid(xlin, xlin, xlin, indexing="ij")

    X = jax.lax.with_sharding_constraint(X, sharding)
    Y = jax.lax.with_sharding_constraint(Y, sharding)
    Z = jax.lax.with_sharding_constraint(Z, sharding)

    # Generate Orszag-Tang initial conditions
    rho, vx, vy, vz, Bx, By, Bz, P = inital_condition(IC, X, Y, Z, gamma, boxsize)

    # Get conserved variables
    Mass, Momx, Momy, Momz, Energy, Bx, By, Bz = get_conserved(rho, vx, vy, vz, P, gamma, Bx, By, Bz)

    # Make animation directory if it doesn't exist
    if not os.path.exists(save_animation_path):
        os.makedirs(save_animation_path, exist_ok=True)

    # Simulation Main Loop
    global_start = time.time() 
    tic = time.time()
    t = 0
    output_counter = 0
    n_iter = 0
    
    while t < t_stop:
        Mass, Momx, Momy, Momz, Energy, dt, rho, Bx, By, Bz = update(
            Mass, Momx, Momy, Momz, Energy, dx, gamma, courant_fac, Bx, By, Bz
        )
        
        rho, vx, vy, vz, P, Bx, By, Bz = get_primitive(Mass, Momx, Momy, Momz, Energy, gamma, Bx, By, Bz)
        # update time
        t += dt
        # update iteration counter
        n_iter += 1
    
    jax.block_until_ready((Mass, Momx, Momy, Momz, Energy, Bx, By, Bz))
    
    # Finalize
    global_end = time.time()
    total_time = global_end - global_start
    mcups = (N**3 * n_iter) / (1e6 * total_time)

    print(f"\nSimulation complete after {n_iter} iterations")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Performance: {mcups:.2f} million cell updates per second (MCUPS)")

if __name__ == "__main__":
    args, config = load_config_and_args()
    main(args, config)