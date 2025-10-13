from modules import *

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