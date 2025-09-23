import jax
import jax.numpy as jnp
from physics import get_primitive
from nonperiodic_boundary_conditions import apply_nonperiodic_boundary_conditions
from thermal_source import apply_thermal_source

# @jax.jit
def update(Mass, Momx, Momy, Energy, dx, dy, gamma, courant_fac, phi, tau, Teq, grav=-1.0, cv=1.5):
    
    rho, vx, vy, P = get_primitive(Mass, Momx, Momy, Energy, phi, gamma)
    
    def extrapolate_to_face_x(f):
        f_XR, f_XL = f , jnp.roll(f, 1, axis=0)
        return f_XL, f_XR
    
    # pas utiliser le roll car non périodique en Y
    def extrapolate_to_face_y(f):
        f_YR, f_YL = f[:, 1:] , f[:, :-1]
        return f_YL, f_YR
        
    c02  = gamma * P / rho
    cmf  = jnp.sqrt(c02)  
    val_max = jnp.maximum(cmf + jnp.abs(vx), cmf + jnp.abs(vy))
    dt = courant_fac * jnp.min(jnp.array([dx, dy])) / jnp.max(val_max)
    
    print("dt:",dt)

    rho_XL, rho_XR,  = extrapolate_to_face_x(rho)
    vx_XL,  vx_XR,  = extrapolate_to_face_x(vx)
    vy_XL,  vy_XR,  = extrapolate_to_face_x(vy)
    P_XL,   P_XR,  = extrapolate_to_face_x(P)
    
    rho_YL, rho_YR = extrapolate_to_face_y(rho)
    vx_YL,  vx_YR  = extrapolate_to_face_y(vx)
    vy_YL,  vy_YR  = extrapolate_to_face_y(vy)
    P_YL,   P_YR   = extrapolate_to_face_y(P)
    
    def get_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma, direction_y=0):
        
        en_L = P_L / (gamma - 1.0) + 0.5 * rho_L * (vx_L**2 + vy_L**2)
        en_R = P_R / (gamma - 1.0) + 0.5 * rho_R * (vx_R**2 + vy_R**2)
        
        My = (rho_L + rho_R) * 0.5 * grav * dy

        c02_L  = gamma * P_L / rho_L
        c02_R  = gamma * P_R / rho_R
        cmfx_L = jnp.sqrt(c02_L)
        cmfx_R = jnp.sqrt(c02_R)

        a_L   = rho_L * cmfx_L
        a_R   = rho_R * cmfx_R
        aface = 1.1 * jnp.maximum(a_L, a_R)

        u_star = 0.5 * (vx_L + vx_R) - 0.5 * (P_R - P_L - My * direction_y) / aface
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

        flux_Energy = jnp.where(
            u_star > 0,
            u_star * en_L + p_star * u_star,
            u_star * en_R + p_star * u_star,
        )

        return flux_Mass, flux_Momx, flux_Momy, flux_Energy

    def apply_fluxes(F, flux_F_X, flux_F_Y, dx, dy, dt):
        
        divX = flux_F_X - jnp.roll(flux_F_X, -1, axis=0)
        F = F.at[:, 1:-1].add((dt/dx) * divX[:, 1:-1])

        divY = (flux_F_Y[:, :-1] - flux_F_Y[:, 1:])  
        F = F.at[:, 1:-1].add((dt/dy) * divY)

        return F

    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, gamma, 1
    )

    Mass   = apply_fluxes(Mass,  flux_Mass_X,   flux_Mass_Y,   dx, dy, dt)
    Momx   = apply_fluxes(Momx,  flux_Momx_X,   flux_Momx_Y,   dx, dy, dt)
    Momy   = apply_fluxes(Momy,  flux_Momy_X,   flux_Momy_Y,   dx, dy, dt)
    Energy = apply_fluxes(Energy,flux_Energy_X, flux_Energy_Y, dx, dy, dt)
    
    # gravity source term 
    Momy = Momy.at[:, 1:-1].add( dt * 0.25 * (2 * Mass[:,1:-1] + Mass[:, 0:-2] + Mass[:, 2:]) * grav)
    
    rho, vx, vy, P = get_primitive(Mass, Momx, Momy, Energy, phi, gamma)
    
    # hermal source term 
    Energy = apply_thermal_source(Energy, rho, vx, vy, phi, cv, Teq, dt, tau)

    # non-periodic boundary conditions
    Mass, Momx, Momy, Energy = apply_nonperiodic_boundary_conditions(
        Mass, Momx, Momy, Energy, rho, vx, vy, phi, gamma, cv, dy
    )

    return Mass, Momx, Momy, Energy, dt, rho
