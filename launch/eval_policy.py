import argparse
import sys

import gym
import numpy as np
import yaml

sys.path.insert(0, ".")
from launch.train_pendulum import pd_policy


def make_policy(cfg):
    """Build the policy callable from config. Extension point for residual policies."""
    kp, kd = cfg["kp"], cfg["kd"]
    print(f"Using kp={kp} and kd={kd}")
    return lambda obs: pd_policy(obs, kp, kd)


def check_success(states, cfg):
    """Check if the last success_hold_steps meet angle and velocity thresholds."""
    hold = cfg["success_hold_steps"]
    if len(states) < hold:
        return False

    tail = np.array(states[-hold:])
    cos_th, sin_th, thdot = tail[:, 0], tail[:, 1], tail[:, 2]
    theta = np.abs(np.arctan2(sin_th, cos_th))  # 0 = upright
    angle_ok = np.all(theta < np.radians(cfg["success_angle_deg"]))
    vel_ok = np.all(np.abs(thdot) < cfg["success_max_thdot"])
    return bool(angle_ok and vel_ok)


def rollout(env, policy, max_steps, cfg):
    """Run one episode, terminating early on success. Returns (states, actions, success)."""
    obs = env.reset()
    states = [obs]
    actions = []
    success = False
    for _ in range(max_steps):
        action = policy(obs)
        obs, _, done, _ = env.step(action)
        states.append(obs)
        actions.append(action)
        if check_success(states, cfg):
            success = True
            break
        if done:
            break
    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32).reshape(-1, 1),
            success)


def compute_trajectory_metrics(states, actions):
    """Compute per-trajectory metrics (absolute values, averaged over steps)."""
    cos_th, sin_th, thdot = states[:, 0], states[:, 1], states[:, 2]

    # Energy: 0.5 * thdot^2 + 10 * (1 - cos_th)  (m=1, l=1, g=10)
    energy = np.abs(0.5 * thdot ** 2 + 10.0 * (1.0 - cos_th))

    return {
        "length": len(actions),
        "energy": float(np.mean(energy)),
        "control_torque": float(np.mean(np.abs(actions))),
        "angular_velocity": float(np.mean(np.abs(thdot))),
    }


def group_stats(metrics_list):
    """Compute mean and std for each metric across a list of per-trajectory dicts."""
    if not metrics_list:
        return {k: {"mean": 0.0, "std": 0.0}
                for k in ["length", "energy", "control_torque", "angular_velocity"]}
    stats = {}
    for key in metrics_list[0]:
        vals = np.array([m[key] for m in metrics_list])
        stats[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return stats


def print_stats_table(results):
    """Print a formatted summary table to stdout."""
    print(f"\nTrajectories: {results['num_trajectories']}  |  "
          f"Success: {results['num_success']}  |  "
          f"Failure: {results['num_failure']}  |  "
          f"Rate: {results['success_rate']:.1%}\n")

    header = f"{'Metric':<20} {'Combined':>20} {'Success':>20} {'Failure':>20}"
    print(header)
    print("-" * len(header))

    for key in ["length", "energy", "control_torque", "angular_velocity"]:
        parts = []
        for group in ["combined", "success", "failure"]:
            m = results[group][key]["mean"]
            s = results[group][key]["std"]
            parts.append(f"{m:8.3f} +/- {s:6.3f}")
        print(f"{key:<20} {parts[0]:>20} {parts[1]:>20} {parts[2]:>20}")
    print()


def evaluate(env, policy, cfg):
    """Run policy evaluation. Prints summary table. Returns (results, all_states, all_actions)."""
    num_traj = cfg["eval_num_trajectories"]
    max_steps = cfg["eval_max_steps"]

    all_states = []
    all_actions = []
    success_metrics = []
    failure_metrics = []
    all_metrics = []

    for i in range(num_traj):
        states, actions, success = rollout(env, policy, max_steps, cfg)
        all_states.append(states)
        all_actions.append(actions)

        metrics = compute_trajectory_metrics(states, actions)
        all_metrics.append(metrics)

        if success:
            success_metrics.append(metrics)
        else:
            failure_metrics.append(metrics)

    results = {
        "num_trajectories": num_traj,
        "num_success": len(success_metrics),
        "num_failure": len(failure_metrics),
        "success_rate": float(len(success_metrics) / num_traj),
        "success": group_stats(success_metrics),
        "failure": group_stats(failure_metrics),
        "combined": group_stats(all_metrics),
    }

    print_stats_table(results)
    return results, all_states, all_actions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate policy on Pendulum")
    parser.add_argument("--config", type=str, default="config/pendulum.yaml")
    parser.add_argument("--output", type=str, default="eval_stats.yaml",
                        help="Path for output stats YAML")
    parser.add_argument("--save-trajectories", type=str, default=None,
                        help="Path to save trajectories as .npz")
    parser.add_argument("--render", action="store_true", default=False,
                        help="Visualize each rollout one at a time")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env = gym.make(cfg["env_name"])
    env.seed(cfg["eval_seed"])
    np.random.seed(cfg["eval_seed"])
    policy = make_policy(cfg)

    if args.render:
        import time
        num_traj = cfg["eval_num_trajectories"]
        max_steps = cfg["eval_max_steps"]
        for i in range(num_traj):
            obs = env.reset()
            states = [obs]
            success = False
            for t in range(max_steps):
                env.render(mode="human")
                action = policy(obs)
                obs, _, done, _ = env.step(action)
                states.append(obs)
                if not success and check_success(states, cfg):
                    success = True
                if done:
                    break
            status = "SUCCESS" if success else "FAIL"
            theta = np.degrees(np.arctan2(obs[1], obs[0]))
            print(f"Trajectory {i+1}/{num_traj}: {status} | "
                  f"steps={len(states)-1} | final_theta={theta:.1f} deg")
            time.sleep(0.5)
        env.close()
    else:
        results, all_states, all_actions = evaluate(env, policy, cfg)
        env.close()

        with open(args.output, "w") as f:
            yaml.dump(results, f, default_flow_style=False, sort_keys=False)
        print(f"Stats saved to {args.output}")

        if args.save_trajectories:
            save_dict = {}
            for i, (s, a) in enumerate(zip(all_states, all_actions)):
                save_dict[f"states_{i}"] = s
                save_dict[f"actions_{i}"] = a
            np.savez(args.save_trajectories, **save_dict)
            print(f"Trajectories saved to {args.save_trajectories}")
