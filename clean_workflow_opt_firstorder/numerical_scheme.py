import jax
import jax.numpy as jnp
from physics import get_primitive

@jax.jit
def update(Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac):
    rho, vx, vy, vz, P = get_primitive(Mass, Momx, Momy, Momz, Energy, gamma)

    def extrapolate_to_face(f):
        f_XR, f_XL = f, jnp.roll(f, 1, axis=0)
        f_YR, f_YL = f, jnp.roll(f, 1, axis=1)
        f_ZR, f_ZL = f, jnp.roll(f, 1, axis=2)
        
        return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR

    c   = jnp.sqrt(gamma * P / rho)
    dtx = dx / jnp.max(jnp.abs(vx) + c)
    dty = dy / jnp.max(jnp.abs(vy) + c)
    dtz = dz / jnp.max(jnp.abs(vz) + c)
    dt  = courant_fac * jnp.minimum(jnp.minimum(dtx, dty), dtz)

    rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = extrapolate_to_face(rho)
    vx_XL, vx_XR, vx_YL, vx_YR, vx_ZL, vx_ZR = extrapolate_to_face(vx)
    vy_XL, vy_XR, vy_YL, vy_YR, vy_ZL, vy_ZR = extrapolate_to_face(vy)
    vz_XL, vz_XR, vz_YL, vz_YR, vz_ZL, vz_ZR = extrapolate_to_face(vz)
    P_XL, P_XR, P_YL, P_YR, P_ZL, P_ZR = extrapolate_to_face(P)

    def get_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, vz_L, vz_R, P_L, P_R, gamma):
        """Calculate fluxes between 2 states with local Lax-Friedrichs/Rusanov rule"""

        # left and right energies
        en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vx_L**2 + vy_L**2 + vz_L**2)
        en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vx_R**2 + vy_R**2 + vz_R**2)

        Pmag_L = P_L
        Pmag_R = P_R

        # find wavespeeds
        c02_L = gamma * P_L / rho_L
        cmfx_L = jnp.sqrt(c02_L)

        c02_R = gamma * P_R / rho_R
        cmfx_R = jnp.sqrt(c02_R)

        a_L = rho_L * cmfx_L
        a_R = rho_R * cmfx_R
        aface = 1.1 * jnp.maximum(a_L,a_R) # renvoie une matrice avec le maximum entre chaque al_ij et ar_ij

        # Define the star states
        u_star = 0.5 * (vx_L + vx_R) - 0.5 * (Pmag_R - Pmag_L) / aface
        p_star = 0.5 * (Pmag_L + Pmag_R) - 0.5 * (vx_R - vx_L) * aface

        v_star = 0.5 * (vy_L + vy_R)
        q_star = - 0.5 * (vy_R - vy_L) * aface

        w_star = 0.5 * (vz_L + vz_R)
        r_star = - 0.5 * (vz_R - vz_L) * aface

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
            u_star * vz_R * rho_R + r_star)
            
        flux_Energy = jnp.where(
            u_star > 0, 
            u_star * en_L + p_star * u_star + q_star * v_star + r_star * w_star, 
            u_star * en_R + p_star * u_star + q_star * v_star + r_star * w_star
        )

        return flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy

    def apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dy, dz, dt):
        F += (dt / dx) * flux_F_X
        F += -(dt / dx) * jnp.roll(flux_F_X, -1, axis=0)
        
        F += (dt / dy) * flux_F_Y
        F += -(dt / dy) * jnp.roll(flux_F_Y, -1, axis=1)
        
        F += (dt / dz) * flux_F_Z
        F += -(dt / dz) * jnp.roll(flux_F_Z, -1, axis=2)
        return F


    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Momz_X, flux_Energy_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, vz_XL, vz_XR, P_XL, P_XR, gamma
    )

    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Momz_Y, flux_Energy_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, vz_YL, vz_YR, P_YL, P_YR, gamma
    )

    flux_Mass_Z, flux_Momz_Z, flux_Momy_Z, flux_Momx_Z, flux_Energy_Z = get_flux(
        rho_ZL, rho_ZR, vz_ZL, vz_ZR, vy_ZL, vy_ZR, vx_ZL, vx_ZR, P_ZL, P_ZR, gamma
    )

    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dy, dz, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dy, dz, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z,  dx, dy, dz, dt)
    Momz = apply_fluxes(Momz, flux_Momz_X, flux_Momz_Y, flux_Momz_Z,  dx, dy, dz, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y,flux_Energy_Z,  dx, dy, dz, dt)

    return Mass, Momx, Momy, Momz, Energy, dt, rho