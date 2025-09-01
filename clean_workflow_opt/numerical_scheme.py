import jax
import jax.numpy as jnp
from physics import get_primitive
from roll import shift_left, shift_right

@jax.jit
def update(Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac, Bx, By, Bz):
    rho, vx, vy, vz, P, Bx, By, Bz = get_primitive(Mass, Momx, Momy, Momz, Energy, gamma, Bx, By, Bz)
    
    U = jnp.stack([rho, P, vx, vy, vz, Bx, By, Bz], axis=0)

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
        f_dx = minmod_1D(shift_left(f, axis=1), f, shift_right(f,axis=1))
        f_dy = minmod_1D(shift_left(f, axis=2), f, shift_right(f,axis=2))
        f_dz = minmod_1D(shift_left(f, axis=3), f, shift_right(f,axis=3))
        return f_dx, f_dy, f_dz
    
    # def get_gradient(f):
    #     """Calculate the gradients of a field"""
    #     f_dx = minmod_1D(jnp.roll(f, 1, axis=1), f, jnp.roll(f, -1, axis=1))
    #     f_dy = minmod_1D(jnp.roll(f, 1, axis=2), f, jnp.roll(f, -1, axis=2))
    #     f_dz = minmod_1D(jnp.roll(f, 1, axis=3), f, jnp.roll(f, -1, axis=3))
    #     return f_dx, f_dy, f_dz

    def extrapolate_to_face(f, f_dx, f_dy, f_dz, dx, dy, dz):
        f_XR, f_XL = f + f_dx * dx / 2, shift_left(f - f_dx * dx / 2, axis=1)
        f_YR, f_YL = f + f_dy * dy / 2, shift_left(f - f_dy * dy / 2, axis=2)
        f_ZR, f_ZL = f + f_dz * dz / 2, shift_left(f - f_dz * dz / 2, axis=3)
      
        return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR
    
    # def extrapolate_to_face(f, f_dx, f_dy, f_dz, dx, dy, dz):
    #     f_XR, f_XL = f + f_dx * dx / 2, jnp.roll(f - f_dx * dx / 2, 1, axis=1)
    #     f_YR, f_YL = f + f_dy * dy / 2, jnp.roll(f - f_dy * dy / 2, 1, axis=2)
    #     f_ZR, f_ZL = f + f_dz * dz / 2, jnp.roll(f - f_dz * dz / 2, 1, axis=3)
        
    #     return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR

    c02 = gamma * P / rho
    ca2 = (Bx**2 + By**2 + Bz**2) / rho
    cap2x = Bx**2 / rho
    cap2y = By**2 / rho
    cap2z = Bz**2 / rho
    cmfx = jnp.sqrt(0.5*(c02+ca2) + 0.5*jnp.sqrt((c02+ca2)**2 - 4*c02*cap2x))
    cmfy = jnp.sqrt(0.5*(c02+ca2) + 0.5*jnp.sqrt((c02+ca2)**2 - 4*c02*cap2y))
    cmfz = jnp.sqrt(0.5*(c02+ca2) + 0.5*jnp.sqrt((c02+ca2)**2 - 4*c02*cap2z))
    val_max = jnp.maximum(jnp.maximum(cmfx + jnp.abs(vx), cmfy + jnp.abs(vy)), cmfz + jnp.abs(vz))
    dt = courant_fac * jnp.min(jnp.array([dx, dy, dz])) / jnp.max(val_max)
    
    Ux, Uy, Uz = get_gradient(U)
    rho_dx, P_dx,  vx_dx, vy_dx, vz_dx, Bx_dx, By_dx, Bz_dx = Ux
    rho_dy, P_dy,  vx_dy, vy_dy, vz_dy, Bx_dy, By_dy, Bz_dy = Uy
    rho_dz, P_dz,  vx_dz, vy_dz, vz_dz, Bx_dz, By_dz, Bz_dz = Uz
    
    div_v = vx_dx + vy_dy + vz_dz
    v = jnp.stack([vx,vy,vz],axis=0)
    B = jnp.stack([Bx,By,Bz],axis=0)
    r = 1.0 / rho
    
    grad_rho = jnp.stack([rho_dx, rho_dy, rho_dz], axis=0)
    grad_p  = jnp.stack([P_dx,   P_dy,   P_dz  ], axis=0)

    grad_vx = jnp.stack([vx_dx,  vx_dy,  vx_dz ], axis=0)
    grad_vy = jnp.stack([vy_dx,  vy_dy,  vy_dz ], axis=0)
    grad_vz = jnp.stack([vz_dx,  vz_dy,  vz_dz ], axis=0)

    grad_Bx = jnp.stack([Bx_dx,  Bx_dy,  Bx_dz ], axis=0)
    B_dx = jnp.stack([Bx_dx,  By_dx,  Bz_dx ], axis=0)
    
    grad_By = jnp.stack([By_dx,  By_dy,  By_dz ], axis=0)
    B_dy = jnp.stack([Bx_dy,  By_dy,  Bz_dy ], axis=0)
    
    grad_Bz = jnp.stack([Bz_dx,  Bz_dy,  Bz_dz ], axis=0)
    B_dz = jnp.stack([Bx_dz,  By_dz,  Bz_dz ], axis=0)
    
    def dot(a, b):
        return jnp.sum(a * b, axis=0)
    
    rho_prime = rho - 0.5 * dt * (rho * div_v + dot(v, grad_rho))
    P_prime = P - 0.5 * dt * (gamma * P * div_v + dot(v, grad_p))
    
    vx_prime = vx - 0.5 * dt * (dot(v, grad_vx) + r * (P_dx + dot(B_dx,B) - dot(grad_Bx,B)))
    vy_prime = vy - 0.5 * dt * (dot(v, grad_vy) + r * (P_dy + dot(B_dy,B) - dot(grad_By,B)))
    vz_prime = vz - 0.5 * dt * (dot(v, grad_vz) + r * (P_dz + dot(B_dz,B) - dot(grad_Bz,B)))
    
    Bx_prime = Bx - 0.5 * dt * (Bx * div_v + dot(v, grad_Bx) - dot(B, grad_vx))
    By_prime = By - 0.5 * dt * (By * div_v + dot(v, grad_By) - dot(B, grad_vy))
    Bz_prime = Bz - 0.5 * dt * (Bz * div_v + dot(v, grad_Bz) - dot(B, grad_vz))
    
    U_prime = jnp.stack([rho_prime, P_prime, vx_prime, vy_prime, vz_prime, Bx_prime, By_prime, Bz_prime], axis=0)

    U_XL, U_XR, U_YL, U_YR, U_ZL, U_ZR = extrapolate_to_face(U_prime, Ux, Uy, Uz, dx, dy, dz)

    rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = U_XL[0], U_XR[0], U_YL[0], U_YR[0], U_ZL[0], U_ZR[0]
    P_XL, P_XR, P_YL, P_YR, P_ZL, P_ZR = U_XL[1], U_XR[1], U_YL[1], U_YR[1], U_ZL[1], U_ZR[1]
    vx_XL, vx_XR, vx_YL, vx_YR, vx_ZL, vx_ZR = U_XL[2], U_XR[2], U_YL[2], U_YR[2], U_ZL[2], U_ZR[2]
    vy_XL, vy_XR, vy_YL, vy_YR, vy_ZL, vy_ZR = U_XL[3], U_XR[3], U_YL[3], U_YR[3], U_ZL[3], U_ZR[3]
    vz_XL, vz_XR, vz_YL, vz_YR, vz_ZL, vz_ZR = U_XL[4], U_XR[4], U_YL[4], U_YR[4], U_ZL[4], U_ZR[4]
    Bx_XL, Bx_XR, Bx_YL, Bx_YR, Bx_ZL, Bx_ZR = U_XL[5], U_XR[5], U_YL[5], U_YR[5], U_ZL[5], U_ZR[5]
    By_XL, By_XR, By_YL, By_YR, By_ZL, By_ZR = U_XL[6], U_XR[6], U_YL[6], U_YR[6], U_ZL[6], U_ZR[6]
    Bz_XL, Bz_XR, Bz_YL, Bz_YR, Bz_ZL, Bz_ZR = U_XL[7], U_XR[7], U_YL[7], U_YR[7], U_ZL[7], U_ZR[7]

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
            u_star * vz_R * rho_R + r_star)
            
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

    def apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dy, dz, dt):
        F += (dt / dx) * flux_F_X
        F += -(dt / dx) * shift_right(flux_F_X, axis=0)
      
        F += (dt / dy) * flux_F_Y
        F += -(dt / dy) * shift_right(flux_F_Y, axis=1)
      
        F += (dt / dz) * flux_F_Z
        F += -(dt / dz) * shift_right(flux_F_Z, axis=2)
        return F

    # def apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dy, dz, dt):
    #     F += (dt / dx) * flux_F_X
    #     F += -(dt / dx) * jnp.roll(flux_F_X, -1, axis=0)
        
    #     F += (dt / dy) * flux_F_Y
    #     F += -(dt / dy) * jnp.roll(flux_F_Y, -1, axis=1)
        
    #     F += (dt / dz) * flux_F_Z
    #     F += -(dt / dz) * jnp.roll(flux_F_Z, -1, axis=2)
    #     return F

    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Momz_X, flux_Energy_X, flux_Bx_X, flux_By_X, flux_Bz_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, vz_XL, vz_XR, P_XL, P_XR, gamma, Bx_XL, Bx_XR, By_XL, By_XR, Bz_XL, Bz_XR
    )

    flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Momz_Y, flux_Energy_Y, flux_By_Y, flux_Bx_Y, flux_Bz_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, vz_YL, vz_YR, P_YL, P_YR, gamma, By_YL, By_YR, Bx_YL, Bx_YR,  Bz_YL, Bz_YR
    )

    flux_Mass_Z, flux_Momz_Z, flux_Momy_Z, flux_Momx_Z, flux_Energy_Z, flux_Bz_Z, flux_By_Z, flux_Bx_Z = get_flux(
        rho_ZL, rho_ZR, vz_ZL, vz_ZR, vy_ZL, vy_ZR, vx_ZL, vx_ZR, P_ZL, P_ZR, gamma, Bz_ZL, Bz_ZR, By_ZL, By_ZR, Bx_ZL, Bx_ZR
    )

    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dy, dz, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dy, dz, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z,  dx, dy, dz, dt)
    Momz = apply_fluxes(Momz, flux_Momz_X, flux_Momz_Y, flux_Momz_Z,  dx, dy, dz, dt)
    Energy = apply_fluxes(Energy, flux_Energy_X, flux_Energy_Y,flux_Energy_Z,  dx, dy, dz, dt)
    Bx = apply_fluxes(Bx, flux_Bx_X, flux_Bx_Y, flux_Bx_Z, dx, dy, dz, dt)
    By = apply_fluxes(By, flux_By_X, flux_By_Y, flux_By_Z, dx, dy, dz, dt)
    Bz = apply_fluxes(Bz, flux_Bz_X, flux_Bz_Y, flux_Bz_Z, dx, dy, dz, dt)

    return Mass, Momx, Momy, Momz, Energy, dt, rho, Bx, By, Bz