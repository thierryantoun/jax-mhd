from modules import *

def inital_condition(IC, X ,Y, Z, gamma):
    if (IC=='convection'):
        
        Nx ,Ny, Nz = X.shape[0], Y.shape[1], Z.shape[2]
        vx   = jnp.zeros((Nx, Ny, Nz+2))
        vy   = jnp.zeros((Nx, Ny, Nz+2))
        vz = jnp.zeros((Nx, Ny, Nz+2))
        rho  = jnp.zeros((Nx, Ny, Nz+2))
        Energy = jnp.zeros((Nx, Ny, Nz+2))
        Teq = jnp.zeros((Nx, Ny, Nz+2))
        phi = jnp.zeros((Nx, Ny, Nz+2))

        grav = -1.0
        cv   = 1.5
        T0   = 10.0
        dTdz = -5.0
        dz = (Z[0,0,1] - Z[0,0,0]) 

        # gravitational potential
        phi_in = - grav * Z  
        phi = phi.at[:, :, 1:-1].set(phi_in)   
        phi = phi.at[:, :, 0].set(phi[:, :, 1] + grav * dz)
        phi = phi.at[:, :, -1].set(phi[:, :, -2] - grav * dz)
        
        k = jnp.arange(Nz+2)[None, None, :]
        Teq   = T0 + dTdz * dz * k

        # bottom boundary condition
        rho = rho.at[:,:,0].set(10.0)
        E_bottom = rho[:,:, 0] * cv * T0 + rho[:, :, 0] * phi[:, :, 0]
        Energy = Energy.at[:, :, 0].set(E_bottom)
        
        # inital conditions
        for k in range(1, Ny + 2):
            Tl = (Energy[:, :, k-1] - rho[:, :, k-1] * phi[:, :, k-1]) / (rho[:, :, k-1] * cv)
            Tr = Tl + dTdz * dz
            
            rhol = rho[:, : ,k-1]
            rhor = rhol * ((gamma -1.) * cv * Tl + 0.5 * grav * dz) / ((gamma -1.) * cv * Tr -0.5 * grav * dz)

            rho  = rho.at[:, :, k].set(rhor)
            E = rhor * cv * Tr + rhor * phi[:, :, k]
            Energy = Energy.at[:, :, k].set(E)
            
        P = (Energy - rho * phi) * (gamma - 1)
                                        
    else:
        print("IC not implemented:", IC)
        sys.exit()

    return rho, vx, vy, vz, P, Teq, phi, Energy