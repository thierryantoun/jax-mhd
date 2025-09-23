import jax.numpy as jnp

def apply_thermal_source(Energy, rho, vx, vy, phi, cv, Teq, dt, tau):
    # Calculate thermal source term
    ec = 0.5 * rho[:, 1:-1] * (vx[:, 1:-1]**2 + vy[:, 1:-1]**2)
    egc = rho[:, 1:-1] * phi[:, 1:-1]
    Tc = (Energy[:, 1:-1] - ec - egc) / (rho[:, 1:-1]*cv)
    Tnew = (Tc + Teq[:, 1:-1] * dt / tau) / (1 + dt / tau)

    Energy = Energy.at[:, 1:-1].set(rho[:, 1:-1] * cv * Tnew + ec + egc)

    return Energy