# A simple example of solving the Euler equations with JAX
# Philip Mocz (2024)

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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
def get_flux_x(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma, Bx_L, Bx_R, By_L, By_R):
    """Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule"""

    # left and right energies
    en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vx_L**2 + vy_L**2) + 0.5 * (Bx_L**2 + By_L**2)
    en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vx_R**2 + vy_R**2) + 0.5 * (Bx_R**2 + By_R**2)

    Pmag_L = P_L + 0.5 * (Bx_L**2 + By_L**2) - Bx_L * Bx_L
    Pmag_R = P_R + 0.5 * (Bx_R**2 + By_R**2) - Bx_R * Bx_R

    Qmag_L = - Bx_L * By_L
    Qmag_R = - Bx_R * By_R

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

    # Define the star states
    u_star = 0.5 * (vx_L + vx_R) - 0.5 * (Pmag_R - Pmag_L) / aface
    p_star = 0.5 * (Pmag_L + Pmag_R) - 0.5 * (vx_R - vx_L) * aface

    v_star = 0.5 * (vy_L + vy_R) - 0.5 * (Qmag_R - Qmag_L)/aface
    q_star = 0.5 * (Qmag_L + Qmag_R) - 0.5 * (vy_R - vy_L) * aface

    # compute fluxes with upwind    
    flux_Mass_X = jnp.where(
        u_star > 0, 
        u_star * rho_L, 
        u_star * rho_R)
    
    flux_Momx_X = jnp.where(
        u_star > 0, 
        u_star * vx_L * rho_L + p_star, 
        u_star * vx_R * rho_R + p_star)

    flux_Momy_X = jnp.where(
        u_star > 0, 
        u_star * vy_L * rho_L + q_star, 
        u_star * vy_R * rho_R + q_star)
        
    flux_Energy_X = jnp.where(
        u_star > 0, 
        u_star * en_L + p_star * u_star + q_star * v_star, 
        u_star * en_R + p_star * u_star + q_star * v_star)

    flux_Bx_X = jnp.where(
        u_star > 0,
        u_star * Bx_L - u_star * Bx_R,
        u_star * Bx_R - u_star * Bx_L
    )

    flux_By_X = jnp.where(
        u_star > 0,
        u_star * By_L - v_star * Bx_R,
        u_star * By_R - v_star * Bx_L
    )

    return flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X, flux_Bx_X, flux_By_X

@jax.jit
def get_flux_y(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma, Bx_L, Bx_R, By_L, By_R):
    """Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule"""

    # left and right energies
    en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vx_L**2 + vy_L**2) + 0.5 * (Bx_L**2 + By_L**2)
    en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vx_R**2 + vy_R**2) + 0.5 * (Bx_R**2 + By_R**2)

    Pmag_L = P_L + 0.5 * (Bx_L**2 + By_L**2) - By_L * By_L
    Pmag_R = P_R + 0.5 * (Bx_R**2 + By_R**2) - By_R * By_R

    Qmag_L = - Bx_L * By_L
    Qmag_R = - Bx_R * By_R

    # find wavespeeds
    c02_L = gamma * P_L / rho_L
    ca2_L = (Bx_L**2 + By_L**2) / rho_L
    cap2y_L = By_L**2 / rho_L
    cmfy_L = jnp.sqrt(0.5*(c02_L+ca2_L)+0.5*jnp.sqrt((c02_L+ca2_L)*(c02_L+ca2_L)-4.*c02_L*cap2y_L))

    c02_R = gamma * P_R / rho_R
    ca2_R = (Bx_R**2 + By_R**2) / rho_R
    cap2y_R = By_R**2 / rho_R
    cmfy_R = jnp.sqrt(0.5*(c02_R+ca2_R)+0.5*jnp.sqrt((c02_R+ca2_R)*(c02_R+ca2_R)-4.*c02_R*cap2y_R))

    a_L = rho_L * cmfy_L
    a_R = rho_R * cmfy_R
    aface = 1.1 * jnp.maximum(a_L, a_R) # renvoie une matrice avec le maximum entre chaque al_ij et ar_ij

    # Define the star states
    u_star = 0.5 * (vy_L + vy_R) - 0.5 * (Pmag_R - Pmag_L) / aface
    p_star = 0.5 * (Pmag_L + Pmag_R) - 0.5 * (vy_R - vy_L) * aface

    v_star = 0.5 * (vx_L + vy_R) - 0.5 * (Qmag_R - Qmag_L)/aface
    q_star = 0.5 * (Qmag_L + Qmag_R) - 0.5 * (vx_R - vx_L) * aface

    # compute fluxes with upwind    
    flux_Mass_y = jnp.where(
        u_star > 0, 
        u_star * rho_L, 
        u_star * rho_R)
    
    flux_Momx_y = jnp.where(
        u_star > 0, 
        u_star * vx_L * rho_L + q_star, 
        u_star * vx_R * rho_R + q_star)

    flux_Momy_y = jnp.where(
        u_star > 0, 
        u_star * vy_L * rho_L + p_star, 
        u_star * vy_R * rho_R + p_star)
        
    flux_Energy_y = jnp.where(
        u_star > 0, 
        u_star * en_L + p_star * u_star + q_star * v_star, 
        u_star * en_R + p_star * u_star + q_star * v_star)

    flux_Bx_y = jnp.where(
        u_star > 0,
        u_star * Bx_L - v_star * By_R,
        u_star * Bx_R - v_star * By_L
    )

    flux_By_y = jnp.where(
        u_star > 0,
        u_star * By_L - u_star * By_R,
        u_star * By_R - u_star * By_L
    )

    return flux_Mass_y, flux_Momx_y, flux_Momy_y, flux_Energy_y, flux_Bx_y, flux_By_y


@jax.jit
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
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X, flux_Bx_X, flux_By_X = get_flux_x(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma, Bx_XL, Bx_XR, By_XL, By_XR
    )
    flux_Mass_Y, flux_Momx_Y, flux_Momy_Y, flux_Energy_Y, flux_Bx_Y, flux_By_Y = get_flux_y(
        rho_YL, rho_YR, vx_YL, vx_YR, vy_YL, vy_YR, P_YL, P_YR, gamma, Bx_YL, Bx_YR, By_YL, By_YR
    )

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
    Bx = apply_fluxes(Bx, flux_Bx_X, flux_Bx_Y, dx, dt)
    By = apply_fluxes(By, flux_By_X, flux_By_Y, dx, dt)

    return Mass, Momx, Momy, Energy, dt, rho, Bx, By

def apply_periodic_boundary(Mass, Momx, Momy, Energy, Bx, By):
    """Apply periodic boundary conditions"""
    
    Mass = Mass.at[0, :].set(Mass[-1, :]) 
    Mass = Mass.at[-1, :].set(Mass[0, :]) 
    Momx = Momx.at[0, :].set(Momx[-1, :])
    Momx = Momx.at[-1, :].set(Momx[0, :])
    Momy = Momy.at[0, :].set(Momy[-1, :])
    Momy = Momy.at[-1, :].set(Momy[0, :])
    Energy = Energy.at[0, :].set(Energy[-1, :])
    Energy = Energy.at[-1, :].set(Energy[0, :])
    Bx = Bx.at[0, :].set(Bx[-1, :])
    Bx = Bx.at[-1, :].set(Bx[0, :])
    By = By.at[0, :].set(By[-1, :])
    By = By.at[-1, :].set(By[0, :])


    Mass = Mass.at[:, 0].set(Mass[:, -1])   
    Mass = Mass.at[:, -1].set(Mass[:, 0])   
    Momx = Momx.at[:, 0].set(Momx[:, -1])
    Momx = Momx.at[:, -1].set(Momx[:, 0])
    Momy = Momy.at[:, 0].set(Momy[:, -1])
    Momy = Momy.at[:, -1].set(Momy[:, 0])
    Energy = Energy.at[:, 0].set(Energy[:, -1])
    Energy = Energy.at[:, -1].set(Energy[:, 0])
    Bx = Bx.at[:, 0].set(Bx[:, -1])
    Bx = Bx.at[:, -1].set(Bx[:, 0])
    By = By.at[:, 0].set(By[:, -1])
    By = By.at[:, -1].set(By[:, 0])

    return Mass, Momx, Momy, Energy, Bx, By


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

    # Generate Blast Initial Conditions
    # center_x, center_y = 0.5 * boxsize, 0.5 * boxsize 
    # radius = 0.2 
    # mask = (X - center_x) ** 2 + (Y - center_y) ** 2 < radius ** 2  

    # rho = jnp.where(mask, 1.0, 1.2)  
    # vx = jnp.zeros_like(X) 
    # vy = jnp.zeros_like(Y) 
    # Bx = vx
    # By = vx
    # P = jnp.where(mask, 10.0 / (gamma - 1), 0.1 / (gamma - 1))

    # Generate Orszag-Tang initial conditions
    rho = jnp.full_like(X, 25.0 / (36.0 * jnp.pi))
    vx = - jnp.sin(2 * jnp.pi * Y)  
    vy = jnp.sin(2 * jnp.pi * X)  
    Bx = -jnp.sin(2 * jnp.pi * Y) / jnp.sqrt(4 * jnp.pi) 
    By = jnp.sin(4 * jnp.pi * X) / jnp.sqrt(4 * jnp.pi) 

    P = jnp.full_like(X, 5.0 / (12.0 * jnp.pi))

    # Generate Brio-Wu initial conditions
    # x_interface = 0.5 * boxsize

    # rho = jnp.where(X < x_interface, 1.0, 0.125)  
    # vx = jnp.zeros_like(X)
    # vy = jnp.zeros_like(X) 
    # Bx = jnp.full_like(X, 0.75) 
    # By = jnp.where(X < x_interface, 1.0, -1.0)
    # P = jnp.where(X < x_interface, 1.0, 0.1)

    # Get conserved variables
    Mass, Momx, Momy, Energy, Bx, By = get_conserved(rho, vx, vy, P, gamma, vol, Bx, By)

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
    save_freq = 1.
    nt = 1000
    # while t < t_stop:
    for it in range(nt):

        time_list.append(t)
        Bx_values.append(Bx[:,N//2])

        # Time step
        Mass, Momx, Momy, Energy, dt, rho, Bx, By = update(
            Mass, Momx, Momy, Energy, vol, dx, gamma, courant_fac, Bx, By
        )

        # boundary conditions
        Mass, Momx, Momy, Energy, Bx, By = apply_periodic_boundary(Mass, Momx, Momy, Energy, Bx, By)

        # determine if we should save the plot
        save_plot = False
        if it > output_counter * save_freq:
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

            # # Print progress
            # print("[it=" + str(n_iter) + " t=" + "{:.6f}".format(t) + "]")
            # print(
            #     "  saved state "
            #     + str(output_counter).zfill(6)
            #     + " of "
            #     + str(int(jnp.ceil(t_stop / save_freq)))
            # )

            # Print million updates per second
            cell_updates = X.shape[0] * X.shape[1] * n_iter
            total_time = time.time() - tic
            mcups = cell_updates / (1e6 * total_time)
            print("  million cell updates / second: ", mcups)

    # jmid = N // 2

    # t_values_np = np.array(time_list)
    # x_values_np = np.array(X)
    # By_values_np = np.array(By[:,jmid])
    # rho_values_np = np.array(rho[:,jmid])

    # plt.figure(1)
    # plt.plot(t_values_np, Bx_values)
    # plt.xlabel("t")
    # plt.ylabel("BX")
    # plt.title("Bx en fonction de t")
    # plt.grid(True)
    # plt.legend()
    # plt.savefig("Bx.png")
    # plt.show()

    # plt.figure(2)
    # plt.plot(x_values_np, By_values_np)
    # plt.xlabel("x")
    # plt.ylabel("Densité By")
    # plt.grid(True)
    # plt.legend()
    # plt.savefig("By.png")
    # plt.show()

    # plt.figure(3)
    # plt.plot(x_values_np, rho_values_np)
    # plt.xlabel("x")
    # plt.ylabel("Densité rho")
    # plt.grid(True)
    # plt.legend()
    # plt.savefig("rho.png")
    # plt.show()

    print("Total time: ", total_time)


if __name__ == "__main__":
    main()
