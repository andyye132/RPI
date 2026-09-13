"""SE(2) Lie group operations in JAX. Pose = (x, y, theta).

Tangent order is (vx, vy, omega). All functions are jit/vmap-safe.
"""
import jax.numpy as jnp


def wrap(a):
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def compose(a, b):
    # a, b: (..., 3) poses. returns a . b
    ca, sa = jnp.cos(a[..., 2]), jnp.sin(a[..., 2])
    x = a[..., 0] + ca * b[..., 0] - sa * b[..., 1]
    y = a[..., 1] + sa * b[..., 0] + ca * b[..., 1]
    th = wrap(a[..., 2] + b[..., 2])
    return jnp.stack([x, y, th], axis=-1)


def inverse(a):
    ca, sa = jnp.cos(a[..., 2]), jnp.sin(a[..., 2])
    x = -(ca * a[..., 0] + sa * a[..., 1])
    y = -(-sa * a[..., 0] + ca * a[..., 1])
    return jnp.stack([x, y, wrap(-a[..., 2])], axis=-1)


def between(a, b):
    # relative pose from a to b: inv(a) . b
    return compose(inverse(a), b)


def log(a):
    # SE(2) logarithm map: pose -> tangent (vx, vy, omega)
    th = wrap(a[..., 2])
    s, c = jnp.sin(th), jnp.cos(th)
    # V^{-1} for SE(2); guard small theta
    half = 0.5 * th
    denom = jnp.where(jnp.abs(th) < 1e-6, 1.0, th)
    a_coef = jnp.where(jnp.abs(th) < 1e-6, 1.0, s / denom)
    b_coef = jnp.where(jnp.abs(th) < 1e-6, 0.0, (1.0 - c) / denom)
    det = a_coef * a_coef + b_coef * b_coef
    vx = (a_coef * a[..., 0] + b_coef * a[..., 1]) / det
    vy = (-b_coef * a[..., 0] + a_coef * a[..., 1]) / det
    return jnp.stack([vx, vy, th], axis=-1)


def exp(v):
    # tangent (vx, vy, omega) -> pose
    th = v[..., 2]
    s, c = jnp.sin(th), jnp.cos(th)
    small = jnp.abs(th) < 1e-6
    denom = jnp.where(small, 1.0, th)
    A = jnp.where(small, 1.0, s / denom)
    B = jnp.where(small, 0.0, (1.0 - c) / denom)
    x = A * v[..., 0] - B * v[..., 1]
    y = B * v[..., 0] + A * v[..., 1]
    return jnp.stack([x, y, wrap(th)], axis=-1)


def retract(pose, delta):
    # right retraction: pose . exp(delta)
    return compose(pose, exp(delta))


def local(pose, target):
    # tangent coordinates of target in pose's frame: log(inv(pose) . target)
    return log(between(pose, target))
