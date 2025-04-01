# A simple example of solving the Euler equations with JAX
# Philip Mocz (2024)

from modules import *
from numerical_scheme import *
from initial_conditions import *  
import configparser


config = configparser.ConfigParser()
config.read("config.ini")

IC = config["simulation"]["IC"]
N = int(config["simulation"]["resolution"])
use_double = config.getboolean("simulation", "double")
boxsize = float(config["simulation"]["boxsize"])
gamma = float(config["simulation"]["gamma"])
courant_fac = float(config["simulation"]["courant_fac"])
t_stop = float(config["simulation"]["t_stop"])
save_freq = float(config["simulation"]["save_freq"])

if use_double:
    print("Using double precision")
    jax.config.update("jax_enable_x64", True)
else:
    print("Using single precision")

def main():
    """Finite Volume simulation"""

    save_animation_path = (
        "output_euler_" + str(N) + ("double" if use_double else "single")
    )

    # Mesh
    dx = boxsize / N
    xlin = jnp.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    X, Y, Z = jnp.meshgrid(xlin, xlin, xlin, indexing="ij")

    # Generate Orszag-Tang initial conditions
    rho, vx, vy, vz, Bx, By, Bz, P = inital_condition(IC, X, Y, Z, gamma, boxsize)

    # Get conserved variables
    Mass, Momx, Momy, Momz, Energy, Bx, By, Bz = get_conserved(rho, vx, vy, vz, P, gamma, Bx, By, Bz)

    # Make animation directory if it doesn't exist
    if not os.path.exists(save_animation_path):
        os.makedirs(save_animation_path, exist_ok=True)

    # Simulation Main Loop
    tic = time.time()
    t = 0
    time_list = []
    Bx_values = []
    output_counter = 0
    n_iter = 0
    save_freq = 0.
    nt = 1000
    while t < t_stop:
    # for it in range(nt):

        # Time step
        Mass, Momx, Momy, Momz, Energy, dt, rho, Bx, By, Bz = update(
            Mass, Momx, Momy, Momz, Energy, dx, gamma, courant_fac, Bx, By, Bz
        )

        # determine if we should save the plot
        save_plot = False
        if t > output_counter * save_freq:
            save_plot = True
            output_counter += 1

        # update time
        t += dt

        # update iteration counter
        n_iter += 1

if __name__ == "__main__":
    main()