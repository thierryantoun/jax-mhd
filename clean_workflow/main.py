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
import jax.numpy as jnp

from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.profiler import trace, StepTraceAnnotation  # <-- profilage
from jax import profiler as jprof

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

    # Allow to visualize how the sharding is done on X
    if jax.process_index() == 0:
        print("X (slice at Z=0):")
        jax.debug.visualize_array_sharding(X[:, :, 0])
        print("X (slice at Y=0):")
        jax.debug.visualize_array_sharding(X[:, 0, :])
        print("X (slice at X=0):")
        jax.debug.visualize_array_sharding(X[0, :, :])

    # Generate Orszag-Tang initial conditions
    rho, vx, vy, vz, Bx, By, Bz, P = inital_condition(IC, X, Y, Z, gamma, boxsize)

    # Get conserved variables
    Mass, Momx, Momy, Momz, Energy, Bx, By, Bz = get_conserved(rho, vx, vy, vz, P, gamma, Bx, By, Bz)

    # Make animation directory if it doesn't exist
    if not os.path.exists(save_animation_path):
        os.makedirs(save_animation_path, exist_ok=True)

    # ---------------------------
    #   CORPS D’ITÉRATION JIT
    # ---------------------------
    from functools import partial
    @partial(jax.jit, static_argnums=(1, 2, 3))  # dx, gamma, courant_fac statiques
    def step_once(state, dx, gamma, courant_fac):
        Mass, Momx, Momy, Momz, Energy, Bx, By, Bz = state
        Mass, Momx, Momy, Momz, Energy, dt, rho, Bx, By, Bz = update(
            Mass, Momx, Momy, Momz, Energy, dx, gamma, courant_fac, Bx, By, Bz
        )
        rho, vx, vy, vz, P, Bx, By, Bz = get_primitive(
            Mass, Momx, Momy, Momz, Energy, gamma, Bx, By, Bz
        )
        return (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz), dt

    # État initial pour la boucle
    state = (Mass, Momx, Momy, Momz, Energy, Bx, By, Bz)

    # --------- TEMPS DE COMPILATION ---------
    t_compile0 = time.perf_counter()
    _compiled = step_once.lower(state, dx, gamma, courant_fac).compile()
    t_compile1 = time.perf_counter()
    print(f"[JIT compile] step_once: {t_compile1 - t_compile0:.3f} s")

    # ------------- EXÉCUTION + TRACE -------------
    global_start = time.time()
    jprof.start_trace("/tmp/jax-trace.json.gz")

    n_iter = 0
    t_phys = 0.0
    for _ in range(5):
        state, dt = _compiled(state)              # <-- on appelle l'exécutable compilé
        dt_val = float(jax.device_get(dt))        # sync ici (mesure propre)
        t_phys += dt_val
        n_iter += 1

    jax.block_until_ready(state)
    jprof.stop_trace()
    global_end = time.time()

    # KPIs
    total_time = global_end - global_start
    mcups = (N**3 * n_iter) / (1e6 * total_time)
    print(f"\nSimulation complete after {n_iter} iterations")
    print(f"Total runtime (exec only): {total_time:.2f} s")
    print(f"Performance: {mcups:.2f} MCUPS")
    print("Trace saved to /tmp/jax-trace.json.gz (drag & drop dans ui.perfetto.dev)")

if __name__ == "__main__":
    args, config = load_config_and_args()
    main(args, config)
