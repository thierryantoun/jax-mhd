import jax
import jax.numpy as jnp
from physics import get_primitive
# from jaxpr import examine_jaxpr

@jax.jit
def update(rho, Momx, Momy, Energy, dx, dy, gamma, courant_fac, Bx, By):
    
    rho, vx, vy, P, Bx, By = get_primitive(rho, Momx, Momy, Energy, gamma, Bx, By)
    
    U_primitive = jnp.stack([rho, P, vx, vy, Bx, By], axis=0)
    
    def minmod_1D(v_l, v_c, v_r):
        dlft = v_c - v_l
        drgt = v_r - v_c
        dcen = 0.5 * (v_r - v_l)
        dsgn = jnp.sign(dcen)
        slop = jnp.minimum(jnp.abs(dlft), jnp.abs(drgt))
        dlim = jnp.where(dlft * drgt < 0, 0.0, slop)
        return dsgn * jnp.minimum(jnp.abs(dcen), dlim)

    def get_gradient(f):
        """Calculate the gradients of a field"""
        f_dx = minmod_1D(jnp.roll(f, 1, axis=1), f, jnp.roll(f, -1, axis=1))
        f_dy = minmod_1D(jnp.roll(f, 1, axis=2), f, jnp.roll(f, -1, axis=2))
        return f_dx, f_dy
    
    def extrapolate_to_face(f, f_dx, f_dy, dx, dy):
        f_XR, f_XL = f + f_dx * dx / 2, jnp.roll(f - f_dx * dx / 2, 1, axis=1)
        f_YR, f_YL = f + f_dy * dy / 2, jnp.roll(f - f_dy * dy / 2, 1, axis=2)
    
        return f_XL, f_XR, f_YL, f_YR

    c02 = gamma * P / rho
    ca2 = (Bx**2 + By**2) / rho
    cap2x = Bx**2 / rho
    cap2y = By**2 / rho
    cmfx = jnp.sqrt(0.5*(c02+ca2) + 0.5*jnp.sqrt((c02+ca2)**2 - 4*c02*cap2x))
    cmfy = jnp.sqrt(0.5*(c02+ca2) + 0.5*jnp.sqrt((c02+ca2)**2 - 4*c02*cap2y))
    val_max = jnp.maximum(cmfx + jnp.abs(vx), cmfy + jnp.abs(vy))
    dt = courant_fac * jnp.min(jnp.array([dx, dy])) / jnp.max(val_max)
        
    Ux, Uy = get_gradient(U_primitive)
    rho_dx, P_dx,  vx_dx, vy_dx, Bx_dx, By_dx = Ux
    rho_dy, P_dy,  vx_dy, vy_dy, Bx_dy, By_dy = Uy
    
    div_v = vx_dx + vy_dy
    v = jnp.stack([vx,vy],axis=0)
    B = jnp.stack([Bx,By],axis=0)
    r = 1.0 / rho
    
    grad_rho = jnp.stack([rho_dx, rho_dy], axis=0)
    grad_p  = jnp.stack([P_dx,   P_dy], axis=0)

    grad_vx = jnp.stack([vx_dx,  vx_dy], axis=0)
    grad_vy = jnp.stack([vy_dx,  vy_dy], axis=0)

    grad_Bx = jnp.stack([Bx_dx,  Bx_dy], axis=0)
    B_dx = jnp.stack([Bx_dx,  By_dx], axis=0)
    
    grad_By = jnp.stack([By_dx,  By_dy], axis=0)
    B_dy = jnp.stack([Bx_dy,  By_dy], axis=0)
    
    def dot(a, b):
        return jnp.sum(a * b, axis=0)
    
    rho_prime = rho - 0.5 * dt * (rho * div_v + dot(v, grad_rho))
    P_prime = P - 0.5 * dt * (gamma * P * div_v + dot(v, grad_p))
    
    vx_prime = vx - 0.5 * dt * (dot(v, grad_vx) + r * (P_dx + dot(B_dx,B) - dot(grad_Bx,B)))
    vy_prime = vy - 0.5 * dt * (dot(v, grad_vy) + r * (P_dy + dot(B_dy,B) - dot(grad_By,B)))
    
    Bx_prime = Bx - 0.5 * dt * (Bx * div_v + dot(v, grad_Bx) - dot(B, grad_vx))
    By_prime = By - 0.5 * dt * (By * div_v + dot(v, grad_By) - dot(B, grad_vy))
    
    U_prime = jnp.stack([rho_prime, P_prime, vx_prime, vy_prime, Bx_prime, By_prime], axis=0)

    U_XL, U_XR, U_YL, U_YR = extrapolate_to_face(U_prime, Ux, Uy, dx, dy)

    rho_XL, rho_XR, rho_YL, rho_YR = U_XL[0], U_XR[0], U_YL[0], U_YR[0]
    P_XL, P_XR, P_YL, P_YR = U_XL[1], U_XR[1], U_YL[1], U_YR[1]
    vx_XL, vx_XR, vx_YL, vx_YR = U_XL[2], U_XR[2], U_YL[2], U_YR[2]
    vy_XL, vy_XR, vy_YL, vy_YR = U_XL[3], U_XR[3], U_YL[3], U_YR[3]
    Bx_XL, Bx_XR, Bx_YL, Bx_YR = U_XL[4], U_XR[4], U_YL[4], U_YR[4]
    By_XL, By_XR, By_YL, By_YR = U_XL[5], U_XR[5], U_YL[5], U_YR[5]

    def get_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma, Bx_L, Bx_R, By_L, By_R):
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

        v_star = 0.5 * (vy_L + vy_R) - 0.5 * (Qmag_R - Qmag_L) / aface
        q_star = 0.5 * (Qmag_L + Qmag_R) - 0.5 * (vy_R - vy_L) * aface

        # compute fluxes with upwind    
        flux_rho = jnp.where(
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
            
        flux_Energy = jnp.where(
            u_star > 0, 
            u_star * en_L + p_star * u_star + q_star * v_star, 
            u_star * en_R + p_star * u_star + q_star * v_star
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

        return flux_rho, flux_Momx, flux_Momy,flux_Energy, flux_Bx, flux_By

    flux_rho_X, flux_Momx_X, flux_Momy_X, flux_Energy_X, flux_Bx_X, flux_By_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma, Bx_XL, Bx_XR, By_XL, By_XR
    )

    flux_rho_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y, flux_By_Y, flux_Bx_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, gamma, By_YL, By_YR, Bx_YL, Bx_YR
    )
    
    sx = dt / dx
    sy = dt / dy
    
    # chaque variable a ses flux propres, batch les opérations ici ne sert à rien
    def apply_fluxes(F, flux_F_X, flux_F_Y, dx, dy, dt):
        F += sx * flux_F_X
        F += - sx * jnp.roll(flux_F_X, -1, axis=0)

        F += sy * flux_F_Y
        F += - sy * jnp.roll(flux_F_Y, -1, axis=1)

        return F
    
    rho = apply_fluxes(rho, flux_rho_X, flux_rho_Y, dx, dy, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dy, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dy, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dy, dt)
    Bx = apply_fluxes(Bx, flux_Bx_X, flux_Bx_Y, dx, dy, dt)
    By = apply_fluxes(By, flux_By_X, flux_By_Y, dx, dy, dt)

    return rho, Momx, Momy, Energy, dt, rho, Bx, By