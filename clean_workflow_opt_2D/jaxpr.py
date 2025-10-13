import jax
import jax.numpy as jnp

def examine_jaxpr(closed_jaxpr):
  jaxpr = closed_jaxpr.jaxpr
  for eqn in jaxpr.eqns:
    print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
  print()
  print("jaxpr:", jaxpr)