import jax
import jax.numpy as jnp
from physics import get_primitive

@jax.jit
def update(Mass, Momx, Momy, Momz, Energy, dx, dy, dz, gamma, courant_fac):

    rho, vx, vy, vz, P = get_primitive(Mass, Momx, Momy, Momz, Energy, gamma)

    # Stack all primitives: shape (5, Nx, Ny, Nz)
    # axis 0 = variable index : [rho, vx, vy, vz, P]
    # axes 1, 2, 3             : spatial X, Y, Z
    U = jnp.stack([rho, vx, vy, vz, P], axis=0)

    c02  = gamma * P / rho
    cmf  = jnp.sqrt(c02)
    val_max = jnp.maximum(jnp.maximum(cmf + jnp.abs(vx), cmf + jnp.abs(vy)),
                          cmf + jnp.abs(vz))
    dt = courant_fac * jnp.min(jnp.array([dx, dy, dz])) / jnp.max(val_max)

    # --- extrapolation vers les faces : un seul appel pour toutes les variables ---
    # Les dimensions spatiales sont maintenant 1, 2, 3 (axe 0 = index variable)
    def extrapolate_to_face(f):
        f_XR, f_XL = f, jnp.roll(f, 1, axis=1)
        f_YR, f_YL = f, jnp.roll(f, 1, axis=2)
        f_ZR, f_ZL = f, jnp.roll(f, 1, axis=3)
        return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR

    U_XL, U_XR, U_YL, U_YR, U_ZL, U_ZR = extrapolate_to_face(U)

    # Déballage des états de face (chaque tableau a la forme (5, Nx, Ny, Nz))
    rho_XL, vx_XL, vy_XL, vz_XL, P_XL = U_XL[0], U_XL[1], U_XL[2], U_XL[3], U_XL[4]
    rho_XR, vx_XR, vy_XR, vz_XR, P_XR = U_XR[0], U_XR[1], U_XR[2], U_XR[3], U_XR[4]
    rho_YL, vx_YL, vy_YL, vz_YL, P_YL = U_YL[0], U_YL[1], U_YL[2], U_YL[3], U_YL[4]
    rho_YR, vx_YR, vy_YR, vz_YR, P_YR = U_YR[0], U_YR[1], U_YR[2], U_YR[3], U_YR[4]
    rho_ZL, vx_ZL, vy_ZL, vz_ZL, P_ZL = U_ZL[0], U_ZL[1], U_ZL[2], U_ZL[3], U_ZL[4]
    rho_ZR, vx_ZR, vy_ZR, vz_ZR, P_ZR = U_ZR[0], U_ZR[1], U_ZR[2], U_ZR[3], U_ZR[4]

    def get_flux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, vz_L, vz_R, P_L, P_R, gamma):

        en_L = P_L / (gamma - 1.0) + 0.5 * rho_L * (vx_L**2 + vy_L**2 + vz_L**2)
        en_R = P_R / (gamma - 1.0) + 0.5 * rho_R * (vx_R**2 + vy_R**2 + vz_R**2)

        Pmag_L = P_L
        Pmag_R = P_R

        c02_L  = gamma * P_L / rho_L
        c02_R  = gamma * P_R / rho_R
        cmfx_L = jnp.sqrt(c02_L)
        cmfx_R = jnp.sqrt(c02_R)

        a_L   = rho_L * cmfx_L
        a_R   = rho_R * cmfx_R
        aface = 1.1 * jnp.maximum(a_L, a_R)

        u_star = 0.5 * (vx_L + vx_R) - 0.5 * (Pmag_R - Pmag_L) / aface
        p_star = 0.5 * (Pmag_L + Pmag_R) - 0.5 * (vx_R - vx_L) * aface

        v_star = 0.5 * (vy_L + vy_R)
        q_star = -0.5 * (vy_R - vy_L) * aface

        w_star = 0.5 * (vz_L + vz_R)
        r_star = -0.5 * (vz_R - vz_L) * aface

        flux_Mass = jnp.where(u_star > 0, u_star * rho_L, u_star * rho_R)

        flux_Momx = jnp.where(
            u_star > 0,
            u_star * vx_L * rho_L + p_star,
            u_star * vx_R * rho_R + p_star,
        )

        flux_Momy = jnp.where(
            u_star > 0,
            u_star * vy_L * rho_L + q_star,
            u_star * vy_R * rho_R + q_star,
        )

        flux_Momz = jnp.where(
            u_star > 0,
            u_star * vz_L * rho_L + r_star,
            u_star * vz_R * rho_R + r_star,
        )

        flux_Energy = jnp.where(
            u_star > 0,
            u_star * en_L + p_star * u_star + q_star * v_star + r_star * w_star,
            u_star * en_R + p_star * u_star + q_star * v_star + r_star * w_star,
        )

        # Retourne le stack [Mass, Mom_normal, Mom_tang1, Mom_tang2, Energy]
        return jnp.stack([flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy], axis=0)

    # Direction X : normale = vx
    # → retourne [flux_Mass, flux_Momx, flux_Momy, flux_Momz, flux_Energy]  ✓
    Flux_X = get_flux(
        rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, vz_XL, vz_XR, P_XL, P_XR, gamma
    )

    # Direction Y : normale = vy ; tangentielles = vx, vz
    # get_flux retourne [Mass, Momy(normal), Momx(tang1), Momz(tang2), Energy]
    # → réindexation [0,2,1,3,4] pour obtenir [Mass, Momx, Momy, Momz, Energy]
    f_Y = get_flux(
        rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, vz_YL, vz_YR, P_YL, P_YR, gamma
    )
    Flux_Y = f_Y[jnp.array([0, 2, 1, 3, 4])]

    # Direction Z : normale = vz ; tangentielles = vy, vx
    # get_flux retourne [Mass, Momz(normal), Momy(tang1), Momx(tang2), Energy]
    # → réindexation [0,3,2,1,4] pour obtenir [Mass, Momx, Momy, Momz, Energy]
    f_Z = get_flux(
        rho_ZL, rho_ZR, vz_ZL, vz_ZR, vy_ZL, vy_ZR, vx_ZL, vx_ZR, P_ZL, P_ZR, gamma
    )
    Flux_Z = f_Z[jnp.array([0, 3, 2, 1, 4])]

    # Stack des variables conservatives : [Mass, Momx, Momy, Momz, Energy]
    U_cons = jnp.stack([Mass, Momx, Momy, Momz, Energy], axis=0)

    sx = dt / dx
    sy = dt / dy
    sz = dt / dz

    def apply_fluxes(F, Fx, Fy, Fz):
        F = F + sx * Fx - sx * jnp.roll(Fx, -1, axis=1)
        F = F + sy * Fy - sy * jnp.roll(Fy, -1, axis=2)
        F = F + sz * Fz - sz * jnp.roll(Fz, -1, axis=3)
        return F

    U_cons = apply_fluxes(U_cons, Flux_X, Flux_Y, Flux_Z)
    Mass, Momx, Momy, Momz, Energy = U_cons[0], U_cons[1], U_cons[2], U_cons[3], U_cons[4]

    return Mass, Momx, Momy, Momz, Energy, dt, rho
