# A simple example of solving the Euler equations with JAX
# Philip Mocz (2024)

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import time
import argparse

save_dir = "result3D"
os.makedirs(save_dir, exist_ok=True)

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
def get_conserved(rho, vx, vy, vz, P, gamma, Bx, By, Bz):
    Mass = rho
    Momx = rho * vx
    Momy = rho * vy
    Momz = rho * vz
    Energy = (P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2 + vz**2) + 0.5 * (Bx**2 + By**2 + Bz**2))
    return Mass, Momx, Momy, Momz, Energy, Bx, By, Bz


@jax.jit
def get_primitive(Mass, Momx, Momy, Momz, Energy, gamma, Bx, By, Bz):
    rho = Mass
    vx = Momx / rho
    vy = Momy / rho
    vz = Momz / rho
    P = (Energy - 0.5 * rho * (vx**2 + vy**2 + vz**2) - 0.5 * (Bx**2 + By**2 + Bz**2)) * (gamma - 1)
    return rho, vx, vy, vz, P, Bx, By, Bz


@jax.jit
def extrapolate_to_face(f):
    f_XR, f_XL = f, jnp.roll(f, 1, axis=0)
    f_YR, f_YL = f, jnp.roll(f, 1, axis=1)
    f_ZR, f_ZL = f, jnp.roll(f, 1, axis=2)
    return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR


@jax.jit
def apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dt):
    F += (dt / dx) * flux_F_X
    F += -(dt / dx) * jnp.roll(flux_F_X, -1, axis=0)
    F += (dt / dx) * flux_F_Y
    F += -(dt / dx) * jnp.roll(flux_F_Y, -1, axis=1)
    F += (dt / dx) * flux_F_Z
    F += -(dt / dx) * jnp.roll(flux_F_Z, -1, axis=2)
    return F

    
@jax.jit
def get_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, vz_L, vz_R, P_L, P_R, gamma, Bx_L, Bx_R, By_L, By_R, Bz_L, Bz_R):
    """Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule"""

    # left and right energies
    en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vx_L**2 + vy_L**2 + vz_L**2) + 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)
    en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vx_R**2 + vy_R**2 + vz_R**2) + 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)

    Pmag_L = P_L + 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2) - Bx_L * Bx_L
    Pmag_R = P_R + 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2) - Bx_R * Bx_R

    Qmag_L = - Bx_L * By_L
    Qmag_R = - Bx_R * By_R

    Rmag_L = -Bx_L * Bz_L
    Rmag_R = -Bx_R * Bz_R

    # find wavespeeds
    c02_L = gamma * P_L / rho_L
    ca2_L = (Bx_L**2 + By_L**2 + Bz_L**2) / rho_L
    cap2x_L = Bx_L**2 / rho_L
    cmfx_L = jnp.sqrt(0.5*(c02_L+ca2_L)+0.5*jnp.sqrt((c02_L+ca2_L)*(c02_L+ca2_L)-4.*c02_L*cap2x_L))

    c02_R = gamma * P_R / rho_R
    ca2_R = (Bx_R**2 + By_R**2 + Bz_R**2) / rho_R
    cap2x_R = Bx_R**2 / rho_R
    cmfx_R = jnp.sqrt(0.5*(c02_R+ca2_R)+0.5*jnp.sqrt((c02_R+ca2_R)*(c02_R+ca2_R)-4.*c02_R*cap2x_R))

    a_L = rho_L * cmfx_L
    a_R = rho_R * cmfx_R
    aface = 1.1 * jnp.maximum(a_L,a_R) # renvoie une matrice avec le maximum entre chaque al_ij et ar_ij

    # Define the star states
    u_star = 0.5 * (vx_L + vx_R) - 0.5 * (Pmag_R - Pmag_L) / aface
    p_star = 0.5 * (Pmag_L + Pmag_R) - 0.5 * (vx_R - vx_L) * aface

    v_star = 0.5 * (vy_L + vy_R) - 0.5 * (Qmag_R - Qmag_L) / aface
    q_star = 0.5 * (Qmag_L + Qmag_R) - 0.5 * (vy_R - vy_L) * aface

    w_star = 0.5 * (vz_L + vz_R) - 0.5 * (Rmag_R - Rmag_L) / aface
    r_star = 0.5 * (Rmag_L + Rmag_R) - 0.5 * (vz_R - vz_L) * aface

    # compute fluxes with upwind    
    flux_Mass = jnp.where(
        u_star > 0, 
        u_star * rho_L, 
        u_star * rho_R)
    
    flux_Momx = jnp.where(
        u_star > 0, 
        u_star * vx_L * rho_L + p_star, 
        u_star * vx_R * rho_R + p_star)

    flux_Momy = jnp.where(
        u_star > 0, 
        u_star * vy_L * rho_L + q_star, 
        u_star * vy_R * rho_R + q_star)

    flux_Momz = jnp.where(
        u_star > 0,
        u_star * vz_L * rho_L + r_star,
        u_star * vz_R * rho_L + r_star)
        
    flux_Energy = jnp.where(
        u_star > 0, 
        u_star * en_L + p_star * u_star + q_star * v_star + r_star * w_star, 
        u_star * en_R + p_star * u_star + q_star * v_star + r_star * w_star
    )

    flux_Bx = jnp.where(
        u_star > 0,
        u_star * Bx_L - u_star * Bx_R,
        u_star * Bx_R - u_star * Bx_L
    )

    flux_By = jnp.where(
        u_star > 0,
        u_star * By_L - v_star * Bx_R,
        u_star * By_R - v_star * Bx_L
    )

    flux_Bz = jnp.where(
        u_star > 0,
        u_star * Bz_L - w_star * Bx_R,
        u_star * Bz_R - w_star * Bx_L
    )

    return flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy, flux_Bx, flux_By, flux_Bz


def update(Mass, Momx, Momy, Momz, Energy, dx, gamma, courant_fac, Bx, By, Bz):

    rho, vx, vy, vz, P, Bx, By, Bz = get_primitive(Mass, Momx, Momy, Momz, Energy, gamma, Bx, By, Bz)
    
    c02 = gamma * P / rho
    ca2 = (Bx**2 + By**2 + Bz**2) / rho
    cap2x, cap2y, cap2z = Bx**2 / rho, By**2 / rho, Bz**2 / rho
    cmfx = jnp.sqrt(0.5*(c02+ca2)+0.5*jnp.sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2x))
    cmfy = jnp.sqrt(0.5*(c02+ca2)+0.5*jnp.sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2y))
    cmfz = jnp.sqrt(0.5*(c02+ca2)+0.5*jnp.sqrt((c02+ca2)*(c02+ca2)-4.*c02*cap2z))
    
    valx = cmfx + jnp.abs(vx)
    valy = cmfy + jnp.abs(vy)
    valz = cmfz + jnp.abs(vz)
    val_max = jnp.maximum(jnp.maximum(valx, valy), valz)
    dt = courant_fac * dx / jnp.max(val_max)

    print("dt:", dt)
    
    rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = extrapolate_to_face(rho)
    vx_XL, vx_XR, vx_YL, vx_YR, vx_ZL, vx_ZR = extrapolate_to_face(vx)
    vy_XL, vy_XR, vy_YL, vy_YR, vy_ZL, vy_ZR = extrapolate_to_face(vy)
    vz_XL, vz_XR, vz_YL, vz_YR, vz_ZL, vz_ZR = extrapolate_to_face(vz)
    P_XL, P_XR, P_YL, P_YR, P_ZL, P_ZR = extrapolate_to_face(P)
    Bx_XL, Bx_XR, Bx_YL, Bx_YR, Bx_ZL, Bx_ZR = extrapolate_to_face(Bx)
    By_XL, By_XR, By_YL, By_YR, By_ZL, By_ZR = extrapolate_to_face(By)
    Bz_XL, Bz_XR, Bz_YL, Bz_YR, Bz_ZL, Bz_ZR = extrapolate_to_face(Bz)

    # compute fluxes
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Momz_X, flux_Energy_X, flux_Bx_X, flux_By_X, flux_Bz_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, vz_XL, vz_XR, P_XL, P_XR, gamma, Bx_XL, Bx_XR, By_XL, By_XR, Bz_XL, Bz_XR
    )

    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Momz_Y, flux_Energy_Y, flux_By_Y, flux_Bx_Y, flux_Bz_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, vz_YL, vz_YR, P_YL, P_YR, gamma, By_YL, By_YR, Bx_YL, Bx_YR,  Bz_YL, Bz_YR
    )

    flux_Mass_Z, flux_Momz_Z, flux_Momy_Z, flux_Momx_Z, flux_Energy_Z, flux_Bz_Z, flux_By_Z, flux_Bx_Z = get_flux(
        rho_ZL, rho_ZR, vz_ZL, vz_ZR, vy_ZL, vy_ZR, vx_ZL, vx_ZR, P_ZL, P_ZR, gamma, Bz_ZL, Bz_ZR, By_ZL, By_ZR, Bx_ZL, Bx_ZR
    )

    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z,  dx, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y,flux_Energy_Z,  dx, dt)
    Bx = apply_fluxes(Bx, flux_Bx_X, flux_Bx_Y, flux_Bx_Z, dx, dt)
    By = apply_fluxes(By, flux_By_X, flux_By_Y, flux_By_Z, dx, dt)
    Bz = apply_fluxes(Bz, flux_Bz_X, flux_Bz_Y, flux_Bz_Z, dx, dt)
    
    return Mass, Momx, Momy, Momz, Energy, dt, rho, Bx, By, Bz


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
    xlin = jnp.linspace(0.5 * dx, boxsize - 0.5 * dx, N)
    X, Y, Z = jnp.meshgrid(xlin, xlin, xlin, indexing="ij")

    # Generate Orszag-Tang initial conditions
    rho = jnp.full_like(X, 25.0 / (36.0 * jnp.pi))
    vx = - jnp.sin(2 * jnp.pi * Y)  
    vx = jnp.full_like(X,0)
    vy = jnp.sin(2 * jnp.pi * X)
    vz = 0.1 * jnp.sin(2 * jnp.pi * Z)
    Bx = -jnp.sin(2 * jnp.pi * Y) / jnp.sqrt(4 * jnp.pi) 
    By = jnp.sin(4 * jnp.pi * X) / jnp.sqrt(4 * jnp.pi) 
    Bz = 0.1 * jnp.sin(2 * jnp.pi * Z) / jnp.sqrt(4 * jnp.pi)
    P = jnp.full_like(X, 5.0 / (12.0 * jnp.pi))

    # Generate Blast Initial Conditions
    # center_x, center_y, center_z = 0.5 * boxsize, 0.5 * boxsize, 0.5 * boxsize
    # radius = 0.2 
    # mask = (X - center_x) ** 2 + (Y - center_y) ** 2 + (Z - center_z) ** 2 < radius ** 2 

    # rho = jnp.where(mask, 1.0, 1.2)  
    # vx = jnp.zeros_like(X) 
    # vy = jnp.zeros_like(Y) 
    # vz = jnp.zeros_like(Z)
    # Bx = vx
    # By = vy
    # Bz = vz
    # P = jnp.where(mask, 10.0 / (gamma - 1), 0.1 / (gamma - 1))

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
        print("t:",t)

        # update iteration counter
        n_iter += 1

if __name__ == "__main__":
    main()
