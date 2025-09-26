import jax
import jax.numpy as jnp
from physics import get_primitive
from nonperiodic_boundary_conditions import apply_nonperiodic_boundary_conditions
from thermal_source import apply_thermal_source

# @jax.jit
def update(Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac, phi, tau, Teq, grav=-1.0, cv=1.5):
    
    rho, vx, vy, vz, P = get_primitive(Mass, Momx, Momy, Momz, Energy, phi, gamma)
    
    def extrapolate_to_face_xy(f):
        f_XR, f_XL = f , jnp.roll(f, 1, axis=0)
        f_YR, f_YL = f , jnp.roll(f, 1, axis=1)
        return f_XL, f_XR, f_YL, f_YR
    
    # pas utiliser le roll car non périodique en Y
    def extrapolate_to_face_z(f):
        f_ZR, f_ZL = f[:, :, 1:] , f[:, :, :-1]
        return f_ZL, f_ZR
        
    cmf = jnp.sqrt(gamma * P / rho)                 
    speed_x = jnp.abs(vx[:, :, 1:-1]) + cmf[:, :, 1:-1]
    speed_y = jnp.abs(vy[:, :, 1:-1]) + cmf[:, :, 1:-1]
    speed_z = jnp.abs(vz[:, :, 1:-1]) + cmf[:, :, 1:-1]
    speed = jnp.maximum(jnp.maximum(speed_x, speed_y), speed_z)

    dt = courant_fac * jnp.min(jnp.array([dx, dy, dz])) / jnp.max(speed)
    
    print("dt:",dt)

    rho_XL, rho_XR, rho_YL, rho_YR  = extrapolate_to_face_xy(rho)
    vx_XL,  vx_XR, vx_YL,  vx_YR = extrapolate_to_face_xy(vx)
    vy_XL,  vy_XR, vy_YL,  vy_YR = extrapolate_to_face_xy(vy)
    vz_XL,  vz_XR, vz_YL,  vz_YR = extrapolate_to_face_xy(vz)
    P_XL,   P_XR, P_YL,   P_YR = extrapolate_to_face_xy(P)
    phi_XL, phi_XR, phi_YL, phi_YR = extrapolate_to_face_xy(phi)
    
    rho_ZL, rho_ZR = extrapolate_to_face_z(rho)
    vx_ZL,  vx_ZR  = extrapolate_to_face_z(vx)
    vy_ZL,  vy_ZR  = extrapolate_to_face_z(vy)
    vz_ZL,  vz_ZR  = extrapolate_to_face_z(vz)
    P_ZL,   P_ZR   = extrapolate_to_face_z(P)
    phi_ZL, phi_ZR = extrapolate_to_face_z(phi)
    
    def get_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, vz_L, vz_R, P_L, P_R, phi_L, phi_R, gamma, direction_z=0):
        
        en_L = P_L / (gamma - 1.0) + 0.5 * rho_L * (vx_L**2 + vy_L**2 + vz_L**2) 
        en_R = P_R / (gamma - 1.0) + 0.5 * rho_R * (vx_R**2 + vy_R**2 + vz_R**2) 
        
        Mz = (rho_L + rho_R) * 0.5 * grav * dz

        c02_L  = gamma * P_L / rho_L
        c02_R  = gamma * P_R / rho_R
        cmfx_L = jnp.sqrt(c02_L)
        cmfx_R = jnp.sqrt(c02_R)

        a_L   = rho_L * cmfx_L
        a_R   = rho_R * cmfx_R
        aface = 1.1 * jnp.maximum(a_L, a_R)

        u_star = 0.5 * (vx_L + vx_R) - 0.5 * (P_R - P_L - Mz * direction_z) / aface
        p_star = 0.5 * (P_L + P_R) - 0.5 * (vx_R - vx_L) * aface

        flux_Mass = jnp.where(u_star > 0, 
            u_star * rho_L, 
            u_star * rho_R,
        )

        flux_Momx = jnp.where(
            u_star > 0,
            u_star * vx_L * rho_L + p_star,
            u_star * vx_R * rho_R + p_star,
        )

        flux_Momy = jnp.where(
            u_star > 0,
            u_star * vy_L * rho_L,
            u_star * vy_R * rho_R,
        )
        
        flux_Momz = jnp.where(
            u_star > 0,
            u_star * vz_L * rho_L,
            u_star * vz_R * rho_R,
        )

        flux_Energy = jnp.where(
            u_star > 0,
            u_star * en_L + p_star * u_star,
            u_star * en_R + p_star * u_star,
        )

        return flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy

    def apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dy, dz, dt):
        
        divX = flux_F_X - jnp.roll(flux_F_X, -1, axis=0)
        F = F.at[:, :, 1:-1].add((dt/dx) * divX[:, :, 1:-1])
        
        divY = flux_F_Y - jnp.roll(flux_F_Y, -1, axis=1)
        F = F.at[:, :, 1:-1].add((dt/dy) * divY[:, :, 1:-1])

        divZ = (flux_F_Z[:, :, :-1] - flux_F_Z[:, :, 1:])  
        F = F.at[:, :, 1:-1].add((dt/dz) * divZ)

        return F

    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Momz_X, flux_Energy_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, vz_XL, vz_XR, P_XL, P_XR, phi_XL, phi_XR, gamma
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Momz_Y, flux_Energy_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, vz_YR, vz_YL, P_YL, P_YR, phi_YL, phi_YR, gamma, 
    )
    
    flux_Mass_Z, flux_Momz_Z, flux_Momy_Z, flux_Momx_Z, flux_Energy_Z = get_flux(
        rho_ZL, rho_ZR, vz_ZL, vz_ZR, vy_ZL, vy_ZR, vx_ZL, vx_ZR, P_ZL, P_ZR, phi_ZL, phi_ZR, gamma, 1
    )
    
    Mass   = apply_fluxes(Mass,  flux_Mass_X,   flux_Mass_Y, flux_Mass_Z,  dx, dy, dz, dt)
    Momx   = apply_fluxes(Momx,  flux_Momx_X,   flux_Momx_Y, flux_Momx_Z,  dx, dy, dz, dt)
    Momy   = apply_fluxes(Momy,  flux_Momy_X,   flux_Momy_Y, flux_Momy_Z,  dx, dy, dz, dt)
    Momz   = apply_fluxes(Momz,  flux_Momz_X,   flux_Momz_Y, flux_Momz_Z,  dx, dy, dz, dt)
    Energy = apply_fluxes(Energy,flux_Energy_X, flux_Energy_Y, flux_Energy_Z, dx, dy, dz, dt)
    
    # gravity source term 
    Momz = Momz.at[:, :, 1:-1].add( dt * 0.25 * (2 * Mass[:, :,1:-1] + Mass[:, :, 0:-2] + Mass[:, :, 2:]) * grav)
    
    rho, vx, vy, vz, P = get_primitive(Mass, Momx, Momy, Momz, Energy, phi, gamma)
    
    # hermal source term 
    Energy = apply_thermal_source(Energy, rho, vx, vy, vz, phi, cv, Teq, dt, tau)

    # non-periodic boundary conditions
    Mass, Momx, Momy, Momz, Energy = apply_nonperiodic_boundary_conditions(
        Mass, Momx, Momy, Momz, Energy, rho, vx, vy, vz, phi, gamma, cv, dz
    )

    return Mass, Momx, Momy, Momz, Energy, dt, rho
