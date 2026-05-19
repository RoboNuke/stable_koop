"""InvertedPendulum-v4 (MuJoCo cartpole) eval logic.

Obs layout (qpos then qvel — gymnasium InvertedPendulum-v4 convention)::

    [cart_x, pole_theta, cart_xdot, pole_theta_dot]

Action: continuous, shape ``(1,)`` — horizontal force on the cart. The
pendulum is passive (no separate actuator), so ``pendulum_control_torque``
is reported as 0 to keep the metric schema parallel with envs that have
an explicit pole actuator.

Success: pole angle within ``success_angle_deg``, ``|pole_theta_dot|``
below ``success_max_thdot``, AND ``|cart_xdot|`` below
``success_max_cart_vel``, all held for ``success_hold_steps`` consecutive
final steps.
"""

from __future__ import annotations

import numpy as np


# Physical parameters of the standard MuJoCo InvertedPendulum-v4 model.
# Used for the energy decomposition only — the env itself doesn't expose
# these to the obs. Adjust here if you customize the XML.
_M_CART = 1.0          # cart mass [kg]
_M_POLE = 0.1          # pole mass [kg]
_POLE_HALF_LEN = 0.3   # distance from joint to pole COM [m]
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
    """True iff the final ``success_hold_steps`` meet angle, |θ̇|, AND |cart_xdot|."""
    hold = cfg["success_hold_steps"]
    if len(states) < hold:
        return False
    tail = np.array(states[-hold:])
    _x, xdot, theta, thdot = _parse_states(tail)
    angle_ok = np.all(np.abs(theta) < np.radians(cfg["success_angle_deg"]))
    angular_vel_ok = np.all(np.abs(thdot) < cfg["success_max_thdot"])
    cart_vel_ok = np.all(np.abs(xdot) < cfg["success_max_cart_vel"])
    return bool(angle_ok and angular_vel_ok and cart_vel_ok)


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
        "pendulum_control_torque": 0.0,  # no direct pole actuator on v4
        "reward": 0.0,  # filled in by policy_rollout
    }
