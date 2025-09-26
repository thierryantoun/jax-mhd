import jax.numpy as jnp

def apply_nonperiodic_boundary_conditions(Mass, Momx, Momy, Momz, Energy, rho, vx, vy, vz, phi, gamma, cv, dz, grav = -1.):
    
    T2 = (Energy[:, :, 2] 
        - 0.5 * Mass[:, :, 2] * (vx[:, :, 2]**2 + 
        vy[:, :, 2]**2 + vz[:, :, 2]**2) - Mass[:, :, 2] * phi[:, :, 2]) / (Mass[:, :, 2] * cv)
    
    T1 = (Energy[:, :, 1] 
        - 0.5 * Mass[:, :, 1] * (vx[:, :, 1]**2 + 
        vy[:, :, 1]**2 + vz[:, :, 1]**2) - Mass[:, :, 1] * phi[:, :, 1]) / (Mass[:, :, 1] * cv)
    
    TO = 2*T1 - T2
    
    Mass = Mass.at[:, :,0].set(Mass[:, :,1]*((gamma-1) * cv * T1 - 0.5 * grav * (dz)) 
            / ((gamma-1) * cv * TO + 0.5 * grav * (dz)))
    
    Momx = Momx.at[:, :,0].set(Mass[:, :,0] * vx[:, :,1])
    Momy = Momy.at[:, :,0].set(Mass[:, :,0] * vy[:, :,1])
    Momz = Momz.at[:, :,0].set( - Mass[:, :,0] * vz[:, :,1])
    
    Energy = Energy.at[:, :,0].set(Mass[:, :,0] * cv * TO + Mass[:, :,0] * phi[:, :,0]
                + 0.5 * Mass[:, :,0] * (vx[:, :,1]**2 + vy[:, :,1]**2 + vz[:, :,1]**2))

    T2 = (Energy[:, :, -3]
      - 0.5 * rho[:, :, -3] * (vx[:, :, -3]**2 + vy[:, :, -3]**2 + vz[:, :, -3]**2)
      - rho[:, :, -3] * phi[:, :, -3]) / (rho[:, :, -3] * cv)

    T1 = (Energy[:, :, -2]
      - 0.5 * rho[:, :, -2] * (vx[:, :, -2]**2 + vy[:, :, -2]**2 + vz[:, :, -2]**2)
      - rho[:, :, -2] * phi[:, :, -2]) / (rho[:, :, -2] * cv)

    T0 = 2.0 * T1 - T2 

    Mass   = Mass.at[:, :, -1].set(Mass[:, :, -2] * (
        ((gamma - 1.0) * cv * T1 + 0.5 * grav * dz) /
        ((gamma - 1.0) * cv * T0 - 0.5 * grav * dz)
    ))
    
    Momx   = Momx.at[:, :, -1].set(Mass[:, :, -1] * vx[:, :, -2])           
    Momy   = Momy.at[:, :, -1].set(Mass[:, :, -1] * vy[:, :, -2])    
    Momz   = Momz.at[:, :, -1].set( - Mass[:, :, -1] * vz[:, :, -2])   
    
    Energy = Energy.at[:, :, -1].set(Mass[:, :, -1] * cv * T0 + Mass[:, :, -1] * phi[:, :, -1] 
            + 0.5 * Mass[:, :, -1] * (vx[:, :, -2]**2 + vy[:, :, -2]**2 + vz[:, :, -2]**2)
    )

    return Mass, Momx, Momy, Momz, Energy
