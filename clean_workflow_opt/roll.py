import jax.lax as lax
import jax.numpy as jnp

def shift_left(f, axis):
    n = f.shape[axis]
    idx = (jnp.arange(n) - 1) % n     
    return jnp.take(f, idx, axis=axis)

def shift_right(f, axis):
    n = f.shape[axis]
    idx = (jnp.arange(n) + 1) % n     
    return jnp.take(f, idx, axis=axis)


