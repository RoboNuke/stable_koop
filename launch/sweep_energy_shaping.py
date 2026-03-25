"""Sweep any 2 parameters of the energy shaping policy, plot success rate heatmap.

k_d is derived from k_p: k_d = 2 * gamma * sqrt(3*k_p - 15) / 3
unless k_d is one of the swept parameters.

Usage examples:
    python -m launch.sweep_energy_shaping --x ke --y kp
    python -m launch.sweep_energy_shaping --x kp --y switch_angle --ke 1.0 --gamma 0.9
    python -m launch.sweep_energy_shaping --x kd --y kp --ke 0.5 --switch-angle 45
"""
import sys
sys.path.insert(0, ".")

import argparse
import math
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

from launch.train_pendulum import energy_shaping_policy
from launch.eval_policy import check_success


PARAM_DEFAULTS = {
    "ke":           (0.1,  3.0,  20),
    "kp":           (5.0,  20.0, 15),
    "kd":           (0.5,  5.0,  15),
    "switch_angle": (10.0, 90.0, 15),  # degrees
    "gamma":        (0.3,  1.0,  15),
}

PARAM_LABELS = {
    "ke":           "k_e (energy shaping gain)",
    "kp":           "k_p (proportional gain)",
    "kd":           "k_d (derivative gain)",
    "switch_angle": "switch angle (degrees)",
    "gamma":        "γ (damping ratio)",
}


def kd_from_kp(kp, gamma):
    """Compute k_d from k_p: k_d = 2 * gamma * sqrt(3*kp - 15) / 3."""
    val = 3.0 * kp - 15.0
    if val <= 0:
        return 0.0
    return 2.0 * gamma * math.sqrt(val) / 3.0


def resolve_params(x_name, x_val, y_name, y_val, fixed):
    """Resolve all policy parameters from the current sweep point + fixed values."""
    params = dict(fixed)
    params[x_name] = x_val
    params[y_name] = y_val

    kp = params["kp"]
    gamma = params["gamma"]

    # Derive kd from kp unless kd is being swept
    if x_name != "kd" and y_name != "kd":
        params["kd"] = kd_from_kp(kp, gamma)

    return params


def run_sweep(cfg, env, x_name, y_name, x_range, y_range, fixed,
              num_traj=200, max_steps=200):
    x_values = np.linspace(*x_range)
    y_values = np.linspace(*y_range)
    results = np.zeros((len(y_values), len(x_values)))

    total = len(y_values) * len(x_values)
    pbar = tqdm(total=total, desc=f"Sweeping {x_name} x {y_name}")
    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            np.random.seed(cfg["eval_seed"])
            p = resolve_params(x_name, x_val, y_name, y_val, fixed)
            successes = 0
            for _ in range(num_traj):
                obs, _ = env.reset()
                states = [obs]
                for t in range(max_steps):
                    action = energy_shaping_policy(
                        obs, kp=p["kp"], kd=p["kd"], k_e=p["ke"],
                        switch_angle=math.radians(p["switch_angle"]),
                    )
                    obs, _, terminated, truncated, _ = env.step(action)
                    states.append(obs)
                    if terminated or truncated:
                        break
                if check_success(states, cfg):
                    successes += 1
            results[i, j] = successes / num_traj
            pbar.update(1)
    pbar.close()
    return x_values, y_values, results


def plot_results(x_values, y_values, results, x_name, y_name, fixed, num_traj):
    fig, ax = plt.subplots(figsize=(10, 7))

    # Convert switch_angle axis to degrees for display
    x_disp = x_values
    y_disp = y_values

    im = ax.pcolormesh(
        x_disp, y_disp, results,
        cmap="RdYlGn", shading="nearest", vmin=0, vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Success Rate")
    ax.set_xlabel(PARAM_LABELS[x_name])
    ax.set_ylabel(PARAM_LABELS[y_name])

    # Build subtitle with fixed params
    swept = {x_name, y_name}
    fixed_strs = []
    for k in ["ke", "kp", "kd", "switch_angle", "gamma"]:
        if k not in swept:
            if k == "kd" and x_name != "kd" and y_name != "kd":
                fixed_strs.append("k_d=f(k_p,γ)")
            elif k == "switch_angle":
                fixed_strs.append(f"switch={fixed[k]}\u00b0")
            elif k == "gamma":
                fixed_strs.append(f"\u03b3={fixed[k]}")
            else:
                fixed_strs.append(f"{k}={fixed[k]}")

    ax.set_title(
        f"Energy Shaping Success Rate ({num_traj} traj)\n"
        f"Fixed: {', '.join(fixed_strs)}"
    )

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"energy_comp_{x_name}_{y_name}_{timestamp}.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {filename}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Sweep 2 energy shaping parameters and plot success rate")
    parser.add_argument("--config", type=str, default="config/pendulum.yaml")
    parser.add_argument("--x", type=str, required=True, choices=PARAM_DEFAULTS.keys(),
                        help="Parameter for x-axis")
    parser.add_argument("--y", type=str, required=True, choices=PARAM_DEFAULTS.keys(),
                        help="Parameter for y-axis")
    parser.add_argument("--x-range", type=float, nargs=3, default=None,
                        metavar=("MIN", "MAX", "N"), help="Override x-axis range")
    parser.add_argument("--y-range", type=float, nargs=3, default=None,
                        metavar=("MIN", "MAX", "N"), help="Override y-axis range")
    parser.add_argument("--num-traj", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)

    # Fixed parameter overrides
    parser.add_argument("--ke", type=float, default=1.0, help="Fixed k_e (default: 1.0)")
    parser.add_argument("--kp", type=float, default=10.0, help="Fixed k_p (default: 10.0)")
    parser.add_argument("--kd", type=float, default=None, help="Fixed k_d (default: derived from kp)")
    parser.add_argument("--switch-angle", type=float, default=60.0, help="Fixed switch angle in degrees (default: 60)")
    parser.add_argument("--gamma", type=float, default=0.8, help="Fixed gamma for kd derivation (default: 0.8)")

    args = parser.parse_args()

    if args.x == args.y:
        parser.error("--x and --y must be different parameters")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Build fixed params dict
    fixed = {
        "ke": args.ke,
        "kp": args.kp,
        "switch_angle": args.switch_angle,
        "gamma": args.gamma,
    }
    # kd: use provided value or derive
    if args.kd is not None:
        fixed["kd"] = args.kd
    else:
        fixed["kd"] = kd_from_kp(args.kp, args.gamma)

    # Sweep ranges
    x_range = tuple(args.x_range) if args.x_range else PARAM_DEFAULTS[args.x]
    y_range = tuple(args.y_range) if args.y_range else PARAM_DEFAULTS[args.y]
    # Convert N to int
    x_range = (x_range[0], x_range[1], int(x_range[2]))
    y_range = (y_range[0], y_range[1], int(y_range[2]))

    print(f"Sweeping {args.x} ({x_range[0]}-{x_range[1]}, n={x_range[2]}) "
          f"x {args.y} ({y_range[0]}-{y_range[1]}, n={y_range[2]})")
    print(f"Fixed: { {k: v for k, v in fixed.items() if k not in {args.x, args.y}} }")

    env = gym.make(cfg["env_name"])
    x_vals, y_vals, results = run_sweep(
        cfg, env, args.x, args.y, x_range, y_range, fixed,
        num_traj=args.num_traj, max_steps=args.max_steps,
    )
    env.close()

    plot_results(x_vals, y_vals, results, args.x, args.y, fixed, args.num_traj)


if __name__ == "__main__":
    main()
