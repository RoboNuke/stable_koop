import argparse
import sys

import gymnasium as gym
import numpy as np
import yaml

sys.path.insert(0, ".")
from launch.train_pendulum import pd_policy, energy_shaping_policy, bang_energy_policy


def zero_policy(obs):
    """Policy that always returns zero action."""
    return np.array([0.0], dtype=np.float32)


def make_policy(cfg):
    """Build the policy callable from config.

    Uses cfg["base_policy"]: "none", "energy", "bang_energy".
    """
    policy_type = cfg.get("base_policy", "none")

    if policy_type == "none":
        print(f"Using base_policy='none' (zero actions)")
        return zero_policy

    kp, kd = cfg["kp"], cfg["kd"]
    ke = cfg["ke"]
    sa = cfg['sa'] * 3.14159 / 180

    if policy_type == "energy":
        print(f"Using base_policy='energy' (kp={kp}, kd={kd}, ke={ke}, sa={sa:.4f})")
        return lambda obs: energy_shaping_policy(obs, kp, kd, ke, switch_angle=sa)
    elif policy_type == "bang_energy":
        print(f"Using base_policy='bang_energy' (kp={kp}, kd={kd}, ke={ke}, sa={sa:.4f})")
        return lambda obs: bang_energy_policy(obs, kp, kd, ke, switch_angle=sa)
    else:
        raise ValueError(f"Unknown base_policy='{policy_type}'. "
                         f"Options: 'none', 'energy', 'bang_energy'")


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
    """Run one episode, terminating early on success. Returns (states, actions, success, total_reward)."""
    obs, _ = env.reset()
    states = [obs]
    actions = []
    total_reward = 0.0
    success = False
    for _ in range(max_steps):
        action = policy(obs) if policy is not None else env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(obs)
        actions.append(action)
        total_reward += reward
        if check_success(states, cfg):
            success = True
            break
        if done:
            break
    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32).reshape(-1, 1),
            success, total_reward)


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
        "reward": 0.0,  # placeholder, filled by evaluate()
    }


def group_stats(metrics_list):
    """Compute mean and std for each metric across a list of per-trajectory dicts."""
    if not metrics_list:
        return {k: {"mean": 0.0, "std": 0.0}
                for k in ["length", "energy", "control_torque", "angular_velocity", "reward"]}
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

    for key in ["length", "energy", "control_torque", "angular_velocity", "reward"]:
        parts = []
        for group in ["combined", "success", "failure"]:
            m = results[group][key]["mean"]
            s = results[group][key]["std"]
            parts.append(f"{m:8.3f} +/- {s:6.3f}")
        print(f"{key:<20} {parts[0]:>20} {parts[1]:>20} {parts[2]:>20}")
    print()


def load_eval_stats(filepath):
    """Load evaluation stats from a YAML file.

    Args:
        filepath: path to an eval_stats.yaml file

    Returns:
        results dict with the same structure as evaluate() returns
    """
    with open(filepath) as f:
        return yaml.safe_load(f)


def make_eval_env(cfg):
    """Create eval env — vectorized if num_parallel_evals > 1, else single."""
    env_name = cfg["env_name"]
    num_envs = cfg.get("num_parallel_evals", 1)
    if num_envs > 1:
        return gym.vector.SyncVectorEnv(
            [lambda en=env_name: gym.make(en) for _ in range(num_envs)]
        )
    return gym.make(env_name)


def _vectorized_evaluate(vec_env, policy, cfg):
    """Run evaluation using a vectorized env in waves.

    Each wave resets all envs, runs up to max_steps, collects one trajectory
    per env. Repeats until num_traj trajectories are collected.

    Returns:
        (all_states, all_actions, all_successes, all_rewards)
    """
    num_traj = cfg["eval_num_trajectories"]
    max_steps = cfg["eval_max_steps"]
    num_envs = vec_env.num_envs

    all_states = []
    all_actions = []
    all_successes = []
    all_rewards = []

    while len(all_states) < num_traj:
        remaining = num_traj - len(all_states)
        active_envs = min(num_envs, remaining)

        obs_batch, _ = vec_env.reset()

        # Per-env buffers
        states = [[obs_batch[i].copy()] for i in range(num_envs)]
        actions = [[] for _ in range(num_envs)]
        rewards = [0.0] * num_envs
        done_flags = [False] * num_envs
        success_flags = [False] * num_envs

        has_batch = hasattr(policy, 'batch') and callable(policy.batch)

        for step in range(max_steps):
            # Batch policy call
            if policy is None:
                action_batch = np.array([vec_env.single_action_space.sample()
                                         for _ in range(num_envs)])
            elif has_batch:
                action_batch = policy.batch(obs_batch)
            else:
                action_batch = np.array([policy(obs_batch[i]) for i in range(num_envs)])

            obs_batch, rew_batch, terminated, truncated, infos = vec_env.step(action_batch)
            dones = terminated | truncated

            for i in range(active_envs):
                if done_flags[i]:
                    continue

                actions[i].append(action_batch[i].copy())
                rewards[i] += float(rew_batch[i])

                if dones[i]:
                    # On auto-reset, obs_batch[i] is the NEW episode obs.
                    # Use the last known obs + action to infer terminal state.
                    # Since Pendulum-v1 truncates (not terminates), the
                    # pre-step obs after applying the action IS the terminal obs.
                    # We record obs_batch[i] from the auto-reset as a proxy —
                    # but for correctness, we skip appending the post-reset obs
                    # and just mark the trajectory as done with what we have.
                    done_flags[i] = True
                    success_flags[i] = check_success(states[i], cfg)
                else:
                    states[i].append(obs_batch[i].copy())
                    if check_success(states[i], cfg):
                        success_flags[i] = True
                        done_flags[i] = True

            if all(done_flags[:active_envs]):
                break

        # Collect from this wave
        for i in range(active_envs):
            if len(all_states) >= num_traj:
                break
            s = np.array(states[i], dtype=np.float32)
            a = np.array(actions[i], dtype=np.float32).reshape(-1, 1) if actions[i] else np.empty((0, 1), dtype=np.float32)
            all_states.append(s)
            all_actions.append(a)
            all_successes.append(success_flags[i])
            all_rewards.append(rewards[i])

    return all_states, all_actions, all_successes, all_rewards


def evaluate(env, policy, cfg):
    """Run policy evaluation. Prints summary table. Returns (results, all_states, all_actions).

    Automatically detects vectorized envs and runs parallel evaluation.
    """
    num_traj = cfg["eval_num_trajectories"]
    is_vectorized = hasattr(env, 'num_envs')

    if is_vectorized:
        all_states, all_actions, all_successes, all_rewards = _vectorized_evaluate(env, policy, cfg)
    else:
        max_steps = cfg["eval_max_steps"]
        all_states, all_actions, all_successes, all_rewards = [], [], [], []
        for i in range(num_traj):
            states, actions, success, total_reward = rollout(env, policy, max_steps, cfg)
            all_states.append(states)
            all_actions.append(actions)
            all_successes.append(success)
            all_rewards.append(total_reward)

    # Compute metrics (same for both paths)
    success_metrics, failure_metrics, all_metrics = [], [], []
    for states, actions, success, reward in zip(all_states, all_actions, all_successes, all_rewards):
        metrics = compute_trajectory_metrics(states, actions)
        metrics["reward"] = float(reward)
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

    np.random.seed(cfg["eval_seed"])
    policy = make_policy(cfg)

    if args.render:
        env = gym.make(cfg["env_name"], render_mode="human")
        import time
        num_traj = cfg["eval_num_trajectories"]
        max_steps = cfg["eval_max_steps"]
        for i in range(num_traj):
            obs, _ = env.reset()
            states = [obs]
            success = False
            for t in range(max_steps):
                env.render()
                action = policy(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
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
        env = make_eval_env(cfg)
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
