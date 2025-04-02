# A simple example of solving the Euler equations with JAX
# Philip Mocz (2024)

import os
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resolution", type=int, default=1024)  # 1024 512 # 256 # 128 # 64
parser.add_argument("--double", action="store_true")
args = parser.parse_args()

if args.double:
    print("Using double precision")
    jax.config.update("jax_enable_x64", True)
else:
    print("Using single precision")


@jax.jit
def get_conserved(rho, vx, vy, P, gamma, vol, Bx, By):
    """Calculate the conserved variables from the primitive variables"""

    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2) + 0.5 * (Bx**2 + By**2)) * vol

    return Mass, Momx, Momy, Energy, Bx, By


@jax.jit
def get_primitive(Mass, Momx, Momy, Energy, gamma, vol, Bx, By):
    """Calculate the primitive variable from the conserved variables"""

    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    P = (Energy / vol - 0.5 * rho * (vx**2 + vy**2) - 0.5 * (Bx**2 + By**2)) * (gamma - 1)

    return rho, vx, vy, P, Bx, By


@jax.jit
def extrapolate_to_face(f):
    """Extrapolate the field from face centers to faces using gradients"""

    f_XL = f  
    f_XR = jnp.roll(f, -1, axis=0) 

    f_YL = f 
    f_YR = jnp.roll(f, -1, axis=1)

    return f_XL, f_XR, f_YL, f_YR


@jax.jit
def apply_fluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """Apply fluxes to conserved variables to update solution state"""

    F += -dt * dx * flux_F_X
    F += dt * dx * jnp.roll(flux_F_X, 1, axis=0)  # left/down roll
    F += -dt * dx * flux_F_Y
    F += dt * dx * jnp.roll(flux_F_Y, 1, axis=1)

    return F


@jax.jit
def get_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma, Bx_L, Bx_R, By_L, By_R):
    """Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule"""

    # left and right energies
    en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vx_L**2 + vy_L**2) + 0.5 * (Bx_L**2 + By_L**2)
    en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vx_R**2 + vy_R**2) + 0.5 * (Bx_R**2 + By_R**2)

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5 * (en_L + en_R)
    Bx_star = 0.5 * (Bx_L + Bx_R)
    By_star = 0.5 * (By_L + By_R)

    P_star = (gamma - 1) * (en_star - 0.5 * (momx_star**2 + momy_star**2) / rho_star)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star
    flux_Momy = momx_star * momy_star / rho_star
    flux_Energy = (en_star + P_star) * momx_star / rho_star
    flux_Bx = 0
    flux_By = momx_star * By_star / rho_star - momy_star * Bx_star / rho_star

    # # find wavespeeds
    # C_L = jnp.sqrt(gamma * P_L / rho_L) + jnp.abs(vx_L)
    # C_R = jnp.sqrt(gamma * P_R / rho_R) + jnp.abs(vx_R)
    # C = jnp.maximum(C_L, C_R)

    # find wavespeeds
    c02_L = gamma * P_L / rho_L
    ca2_L = (Bx_L**2 + By_L**2) / rho_L
    cap2x_L = Bx_L**2 / rho_L
    cmfx_L = jnp.sqrt(0.5*(c02_L+ca2_L)+0.5*jnp.sqrt((c02_L+ca2_L)*(c02_L+ca2_L)-4.*c02_L*cap2x_L))

    c02_R = gamma * P_R / rho_R
    ca2_R = (Bx_R**2 + By_R**2) / rho_R
    cap2x_R = Bx_R**2 / rho_R
    cmfx_R = jnp.sqrt(0.5*(c02_R+ca2_R)+0.5*jnp.sqrt((c02_R+ca2_R)*(c02_R+ca2_R)-4.*c02_R*cap2x_R))

    a_L = rho_L * cmfx_L
    a_R = rho_R * cmfx_R
    aface = 1.1 * jnp.maximum(a_L,a_R) # renvoie une matrice avec le maximum entre chaque al_ij et ar_ij

    # add stabilizing diffusive term
    flux_Mass -= aface * 0.5 * (rho_R - rho_L)
    flux_Momx -= aface * 0.5 * (rho_R * vx_R - rho_L * vx_L)
    flux_Momy -= aface * 0.5 * (rho_R * vy_R - rho_L * vy_L)
    flux_Energy -= aface * 0.5 * (en_R - en_L)
    flux_Bx -=  aface * 0.5 * (Bx_R - Bx_L)
    flux_By -= aface * 0.5 * (By_R - By_L)

    return flux_Mass, flux_Momx, flux_Momy, flux_Energy, flux_Bx, flux_By


def update(Mass, Momx, Momy, Energy, vol, dx, gamma, courant_fac, Bx, By):
    """Take a simulation timestep"""

    # get Primitive variables
    rho, vx, vy, P, Bx, By = get_primitive(Mass, Momx, Momy, Energy, gamma, vol, Bx, By)

    # get time step (CFL) = dx / max signal speed
    c02 = gamma * P / rho
    ca2 = (Bx**2 + By**2) / rho
    cap2x = Bx**2 / rho
    cap2y = By**2 / rho
    cmfx = jnp.sqrt(0.5*(c02+ca2)+0.5*jnp.sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2x))
    cmfy = jnp.sqrt(0.5*(c02+ca2)+0.5*jnp.sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2y))

    valx = cmfx + jnp.abs(vx)
    valy = cmfy + jnp.abs(vy)
    val_max = jnp.maximum(valx,valy)

    dt = courant_fac * dx / jnp.max(val_max)

    print("dt:",dt)

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR = extrapolate_to_face(rho)
    vx_XL, vx_XR, vx_YL, vx_YR = extrapolate_to_face(vx)
    vy_XL, vy_XR, vy_YL, vy_YR = extrapolate_to_face(vy)
    P_XL, P_XR, P_YL, P_YR = extrapolate_to_face(P)
    Bx_XL, Bx_XR, Bx_YL, Bx_YR = extrapolate_to_face(Bx)
    By_XL, By_XR, By_YL, By_YR = extrapolate_to_face(By)

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X, flux_Bx_X, flux_By_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma, Bx_XL, Bx_XR, By_XL, By_XL
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y, flux_By_Y, flux_Bx_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, gamma, By_YL, By_YL, Bx_YL, Bx_YR
    )

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
    Bx = apply_fluxes(Bx, flux_Bx_X, flux_Bx_Y, dx, dt)
    By = apply_fluxes(By, flux_By_X, flux_By_Y, dx, dt)

    return Mass, Momx, Momy, Energy, dt, rho, Bx, By


def main():
    """Finite Volume simulation"""

    # Simulation parameters
    N = args.resolution
    boxsize = 1.0
    gamma = 5./3.  # ideal gas gamma
    courant_fac = 0.4
    t_stop = 0.5
    save_freq = 0.1
    save_animation_path = (
        "output_euler_" + str(N) + ("double" if args.double else "single")
    )

    # Mesh
    dx = boxsize / N
    vol = dx**2
    xlin = jnp.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    X, Y = jnp.meshgrid(xlin, xlin, indexing="ij")

    # Generate Initial Conditions - opposite moving streams with perturbation
    # w0 = 0.1
    # sigma = 0.05 / jnp.sqrt(2.0)
    # rho = 1.0 + (jnp.abs(Y - 0.5) < 0.25)
    # vx = -0.5 + (jnp.abs(Y - 0.5) < 0.25)
    # vy = (
    #     w0
    #     * jnp.sin(4 * jnp.pi * X)
    #     * (
    #         jnp.exp(-((Y - 0.25) ** 2) / (2 * sigma**2))
    #         + jnp.exp(-((Y - 0.75) ** 2) / (2 * sigma**2))
    #     )
    # )
    # P = 2.5 * jnp.ones(X.shape)

    # Generate Orszag-Tang initial conditions
    rho = jnp.full_like(X, 25.0 / (36.0 * jnp.pi))
    vx = - jnp.sin(2 * jnp.pi * Y)  
    vy = jnp.sin(2 * jnp.pi * X)  
    Bx = -jnp.sin(2 * jnp.pi * Y) / jnp.sqrt(4 * jnp.pi) 
    By = jnp.sin(4 * jnp.pi * X) / jnp.sqrt(4 * jnp.pi) 

    P = jnp.full_like(X, 5.0 / (12.0 * jnp.pi))

    # Get conserved variables
    Mass, Momx, Momy, Energy, Bx, By = get_conserved(rho, vx, vy, P, gamma, vol, Bx, By)

    # Make animation directory if it doesn't exist
    if not os.path.exists(save_animation_path):
        os.makedirs(save_animation_path, exist_ok=True)

    # Simulation Main Loop
    # Simulation Main Loop
    tic = time.time()
    t = 0
    output_counter = 0
    n_iter = 0
    save_freq = 1.
    nt = 1000
    t_stop = 0.5
    while t < t_stop:

        # Time step
        Mass, Momx, Momy, Energy, dt, rho, Bx, By = update(
            Mass, Momx, Momy, Energy, vol, dx, gamma, courant_fac, Bx, By
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

        # save plot
        if save_plot:
            plt.imsave(
                save_animation_path + "/rho" + str(output_counter).zfill(6) + ".png",
                jnp.rot90(rho),
                cmap="jet",
                vmin=0.8,
                vmax=2.2,
            )

            # Print progress
            print("[it=" + str(n_iter) + " t=" + "{:.6f}".format(t) + "]")
            print(
                "  saved state "
                + str(output_counter).zfill(6)
                + " of "
                + str(int(jnp.ceil(t_stop / save_freq)))
            )

            # Print million updates per second
            cell_updates = X.shape[0] * X.shape[1] * n_iter
            total_time = time.time() - tic
            mcups = cell_updates / (1e6 * total_time)
            print("  million cell updates / second: ", mcups)

    print("Total time: ", total_time)


if __name__ == "__main__":
    main()
