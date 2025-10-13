from modules import *

def inital_condition(IC, X ,Y, gamma, boxsize):
    
    if (IC=='orszag-tang3D'):
        rho = jnp.full_like(X, 25.0 / (36.0 * jnp.pi))
        vx = - jnp.sin(2 * jnp.pi * Y)  
        vy = jnp.sin(2 * jnp.pi * X)
        Bx = -jnp.sin(2 * jnp.pi * Y) / jnp.sqrt(4 * jnp.pi) 
        By = jnp.sin(4 * jnp.pi * X) / jnp.sqrt(4 * jnp.pi) 
        P = jnp.full_like(X, 5.0 / (12.0 * jnp.pi))
    
    elif (IC=='blast'):
        center_x, center_y = 0.5 * boxsize, 0.5 * boxsize
        radius = 0.1 
        distance2 = (X - center_x)**2 + (Y - center_y)**2
        mask = distance2 < radius**2  
        rho = jnp.where(mask, 1.0, 1.2)
        vx = jnp.zeros_like(X)
        vy = jnp.zeros_like(Y)
        Bx = jnp.zeros_like(X)
        By = jnp.zeros_like(Y)
        P = jnp.where(mask, 10.0 / (gamma - 1), 0.1 / (gamma - 1))
    else:
        print("IC not implemented:", IC)
        sys.exit()

    return rho, vx, vy, Bx, By, P