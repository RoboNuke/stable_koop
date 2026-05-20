"""InvertedPendulum-v4 (MuJoCo cartpole) eval logic.

Obs layout (qpos then qvel — gymnasium InvertedPendulum-v4 convention)::

    [cart_x, pole_theta, cart_xdot, pole_theta_dot]

Action: continuous, shape ``(1,)`` — horizontal force on the cart. The
pendulum is passive (no separate actuator), so only ``cart_control`` is
reported; there is no pole-torque channel to summarize.

Success: the trajectory survived to the rollout time-limit (the env's
own ``|pole_theta| > 0.2`` termination was not triggered). This is the
natural "stayed balanced for the whole episode" metric for this task;
the tighter angle / velocity / cart-velocity tail check from the
previous version was too aggressive for pure linear feedback policies.
"""

from __future__ import annotations

import numpy as np


# Physical parameters of the MuJoCo InvertedPendulum-v4/v5 model, read
# from `env.unwrapped.model.body_mass` and the pole-capsule geometry. The
# env itself doesn't expose these to the obs, so they live here as
# constants — adjust if you customize the XML.
_M_CART = 10.471976    # cart body mass [kg]
_M_POLE = 5.018592     # pole body mass [kg]
_POLE_HALF_LEN = 0.3   # distance from hinge to pole COM [m] (capsule half-length)
_G = 9.81              # gravitational acceleration [m/s^2]


def _parse_states(states):
    """Return ``(cart_x, cart_xdot, theta, theta_dot)`` columns from ``states``.

    Note the swap from the raw obs layout: gym packs the obs as
    ``[cart_x, theta, cart_xdot, theta_dot]`` (qpos then qvel), but every
    downstream consumer in this module wants the (pose, velocities) order
    grouped sensibly, so we reorder here once.
    """
    return (
        states[..., 0],  # cart_x
        states[..., 2],  # cart_xdot (qvel[0])
        states[..., 1],  # theta (qpos[1])
        states[..., 3],  # theta_dot (qvel[1])
    )


def inverted_pendulum_check_success(states, cfg):
    """True iff the rollout reached its time limit without env termination.

    InvertedPendulum-v5 terminates the episode the moment ``|theta| > 0.2``
    rad, so a trajectory of length ``eval_max_steps`` (i.e. one initial
    obs + ``eval_max_steps`` step obses) means the policy never let the
    pole fall — that *is* the success criterion. Angle / velocity
    tolerances in ``cfg`` are ignored on purpose.
    """
    return len(states) >= cfg["max_steps"] + 1


def inverted_pendulum_compute_metrics(states, actions):
    """Per-trajectory mean-absolute metrics + energy decomposition."""
    _x, xdot, theta, thdot = _parse_states(states)

    # Cart kinetic energy.
    cart_kin = 0.5 * _M_CART * xdot ** 2

    # Pole energy (pole alone, treated as point mass at half-length).
    # KE includes both the cart-frame rotational contribution and the cart's
    # translation; PE = m g L cos(theta) (positive when upright, matches the
    # Pendulum-v1 sign convention).
    v_com_x = xdot + _POLE_HALF_LEN * np.cos(theta) * thdot
    v_com_y = -_POLE_HALF_LEN * np.sin(theta) * thdot
    pole_kin = 0.5 * _M_POLE * (v_com_x ** 2 + v_com_y ** 2)
    pole_pot = _M_POLE * _G * _POLE_HALF_LEN * np.cos(theta)
    pole_energy = pole_kin + pole_pot

    total_energy = cart_kin + pole_energy

    # Action[0] is the cart force; there is no separate pole actuator on v4.
    cart_force = np.abs(actions[..., 0])

    return {
        "length": len(actions),
        "cart_velocity": float(np.mean(np.abs(xdot))),
        "angular_velocity": float(np.mean(np.abs(thdot))),
        "cart_energy": float(np.mean(cart_kin)),
        "pendulum_energy": float(np.mean(pole_energy)),
        "total_energy": float(np.mean(total_energy)),
        "cart_control": float(np.mean(cart_force)),
        "reward": 0.0,  # filled in by rollout
    }
