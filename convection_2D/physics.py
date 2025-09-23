from modules import *

@jax.jit
def get_conserved(rho, vx, vy, P, phi, gamma):
    Mass = rho
    Momx = rho * vx
    Momy = rho * vy
    Energy = (P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2) + rho * phi)
    return Mass, Momx, Momy, Energy


@jax.jit
def get_primitive(Mass, Momx, Momy, Energy, phi, gamma):
    rho = Mass
    vx = Momx / rho
    vy = Momy / rho
    P = (Energy - 0.5 * rho * (vx**2 + vy**2) - rho * phi) * (gamma - 1)
    return rho, vx, vy, P