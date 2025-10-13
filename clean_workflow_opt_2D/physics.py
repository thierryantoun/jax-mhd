from modules import *

@jax.jit
def get_conserved(rho, vx, vy, P, gamma, Bx, By):
    Mass = rho
    Momx = rho * vx
    Momy = rho * vy
    Energy = (P / (gamma - 1) + 0.5 * rho * (vx**2 + vy**2) + 0.5 * (Bx**2 + By**2))
    return Mass, Momx, Momy, Energy, Bx, By


@jax.jit
def get_primitive(Mass, Momx, Momy, Energy, gamma, Bx, By):
    rho = Mass
    vx = Momx / rho
    vy = Momy / rho
    P = (Energy - 0.5 * rho * (vx**2 + vy**2) - 0.5 * (Bx**2 + By**2)) * (gamma - 1)
    return rho, vx, vy, P, Bx, By