from modules import *

def inital_condition(IC, X ,Y, gamma, boxsize):
    
    if (IC=='orszag-tang3D'):
        rho = jnp.full_like(X, 25.0 / (36.0 * jnp.pi))
        vx = - jnp.sin(2 * jnp.pi * Y)  
        vy = jnp.sin(2 * jnp.pi * X)
        P = jnp.full_like(X, 5.0 / (12.0 * jnp.pi))
        Teq, phi, Energy = jnp.zeros_like(X), jnp.zeros_like(X), jnp.zeros_like(X)
        
    if (IC=='blast'):
        center_x, center_y= 0.5 * boxsize, 0.5 * boxsize
        radius = 0.1 
        distance2 = (X - center_x)**2 + (Y - center_y)**2
        mask = distance2 < radius**2  
        rho = jnp.where(mask, 1.0, 1.2)
        vx = jnp.zeros_like(X)
        vy = jnp.zeros_like(Y)
        P = jnp.where(mask, 10.0 / (gamma - 1), 0.1 / (gamma - 1))
        Teq, phi, Energy = jnp.zeros_like(X), jnp.zeros_like(X), jnp.zeros_like(X)
        
    elif (IC=='convection'):
        
        Nx ,Ny = X.shape[0], Y.shape[1]
        vx   = jnp.zeros((Nx, Ny+2))
        vy   = jnp.zeros((Nx, Ny+2))
        rho  = jnp.zeros((Nx, Ny+2))
        Energy = jnp.zeros((Nx, Ny+2))
        Teq = jnp.zeros((Nx, Ny+2))
        phi = jnp.zeros((Nx, Ny+2))

        grav = -1.0
        cv   = 1.5
        T0   = 10.0
        dTdy = -5.0
        dy = (Y[0,1] - Y[0,0]) 

        # gravitational potential
        phi_in = - grav * Y  
        phi = phi.at[:, 1:-1].set(phi_in)   
        phi = phi.at[:, 0].set(phi[:, 1] + grav * dy)
        phi = phi.at[:, -1].set(phi[:, -2] - grav * dy)
        
        j_idx = jnp.arange(Ny+2)[None, :]
        Teq   = T0 + dTdy * dy * j_idx

        # bottom boundary condition
        rho = rho.at[:,0].set(10.0)
        E_bottom = rho[:, 0] * cv * T0 + rho[:, 0] * phi[:, 0]
        Energy = Energy.at[:, 0].set(E_bottom)
        
        # inital conditions
        for j in range(1, Ny + 2):
            Tl = (Energy[:, j-1] - rho[:, j-1] * phi[:, j-1]) / (rho[:, j-1] * cv)
            Tr = Tl + dTdy * dy
            
            rhol = rho[:, j-1]
            rhor = rhol * ((gamma -1.) * cv * Tl + 0.5 * grav * dy) / ((gamma -1.) * cv * Tr -0.5 * grav * dy)

            rho  = rho.at[:, j].set(rhor)
            E = rhor * cv * Tr + rhor * phi[:, j]
            Energy = Energy.at[:, j].set(E)
            
        P = (Energy - rho * phi) * (gamma - 1)
                                        
    else:
        print("IC not implemented:", IC)
        sys.exit()

    return rho, vx, vy, P, Teq, phi, Energy