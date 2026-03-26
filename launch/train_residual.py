"""Train a residual policy using SKRL SAC.

The residual policy outputs z_ref (a latent-space reference vector). The
residual control action is computed via LQR: u_res = F @ z_ref. The total
action applied to the environment is: clip(base_action + u_res, env_bounds).

Usage:
    python -m launch.train_residual --config config/pendulum.yaml
"""
import argparse
import os
import sys

sys.path.insert(0, ".")

import gymnasium as gym
import numpy as np
import torch
import yaml

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env

from launch.eval_policy import evaluate as evaluate_policy, load_eval_stats, make_eval_env
from launch.train_pendulum import energy_shaping_policy
from model.residual import StochasticActor, Critic
from wrappers.residual import ResidualPolicyEnv


# ---------------------------------------------------------------------------
# Composite policy
# ---------------------------------------------------------------------------

def make_composite_policy(base_policy, residual_model, lqr_F_np, z_ref_limit, device, action_bounds):
    """Build a composite policy: base + LQR residual from learned z_ref.

    Supports both single obs and batched obs (for vectorized eval).
    Single: policy(obs) -> action  (obs shape: (obs_dim,))
    Batched: policy.batch(obs_batch) -> action_batch  (obs shape: (N, obs_dim))

    Args:
        base_policy: callable(obs) -> base_action
        residual_model: SKRL actor model (outputs raw actions in [-1, 1])
        lqr_F_np: numpy array (action_dim, latent_dim)
        z_ref_limit: float, scaling from raw action to z_ref
        device: torch device
        action_bounds: (low, high) numpy arrays for clipping

    Returns:
        callable with .batch method for vectorized eval
    """
    act_low, act_high = action_bounds

    def policy(obs):
        base_action = base_policy(obs)
        obs_aug = np.concatenate([obs, base_action]).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_aug).unsqueeze(0).to(device)
            raw_action = residual_model.act({"states": obs_t})[0]
            raw_action_np = raw_action.cpu().numpy().flatten()
        z_ref = z_ref_limit * raw_action_np
        u_res = lqr_F_np @ z_ref
        return np.clip(base_action + u_res, act_low, act_high)

    def batch_policy(obs_batch):
        """Batched composite policy — single torch forward pass for all envs."""
        N = len(obs_batch)
        base_actions = np.array([base_policy(obs_batch[i]) for i in range(N)])
        obs_aug = np.concatenate([obs_batch, base_actions], axis=-1).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_aug).to(device)
            raw_actions = residual_model.act({"states": obs_t})[0]
            raw_actions_np = raw_actions.cpu().numpy()
        z_refs = z_ref_limit * raw_actions_np
        u_res = z_refs @ lqr_F_np.T  # (N, action_dim)
        total = base_actions + u_res
        return np.clip(total, act_low, act_high)

    policy.batch = batch_policy
    return policy


# ---------------------------------------------------------------------------
# Evaluation callback
# ---------------------------------------------------------------------------

def run_eval(residual_model, base_policy, lqr_F, z_ref_limit, cfg, eval_env, writer, step, device):
    """Evaluate composite policy and log to TensorBoard."""
    residual_model.eval()
    # Use single_action_space for vectorized envs, action_space for single envs
    act_space = eval_env.single_action_space if hasattr(eval_env, 'num_envs') else eval_env.action_space
    action_bounds = (act_space.low, act_space.high)
    policy = make_composite_policy(base_policy, residual_model, lqr_F, z_ref_limit, device, action_bounds)
    results, _, _ = evaluate_policy(eval_env, policy, cfg)

    writer.add_scalar("eval_total_metric/success_rate", results["success_rate"], step)
    group_map = {"combined": "total", "success": "success", "failure": "failure"}
    metrics = ["length", "energy", "control_torque", "angular_velocity", "reward"]
    for group, label in group_map.items():
        for key in metrics:
            writer.add_scalar(f"eval_{label}_metric/{key}", results[group][key]["mean"], step)
            writer.add_scalar(f"eval_{label}_metric_std/{key}", results[group][key]["std"], step)

    residual_model.train()
    return results


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_residual(base_policy, lqr, cfg, run_dir, z_ref_limit=1.0, keep_all_ckpts=False):
    """Train residual policy with SAC.

    The SAC agent outputs z_ref (latent space reference). The residual
    control action is u_res = F @ z_ref, added to the base policy action.

    Args:
        base_policy: callable(obs) -> action
        lqr: LQR object with .F attribute (action_dim, latent_dim)
        cfg: config dict
        run_dir: output directory

    Returns:
        Trained SKRL actor model.
    """
    phase_dir = os.path.join(run_dir, "residual_train")
    os.makedirs(phase_dir, exist_ok=True)
    ckpt_dir = os.path.join(phase_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    latent_dim = cfg["latent_dim"]
    num_envs = cfg.get("residual_num_envs", 8)
    total_timesteps = cfg.get("residual_total_timesteps", 100000)
    eval_interval = cfg.get("residual_eval_interval", 5000)
    actor_hidden_size = cfg.get("residual_actor_hidden_size", 64)
    actor_hidden_layers = cfg.get("residual_actor_hidden_layers", 2)
    critic_hidden_size = cfg.get("residual_critic_hidden_size", 64)
    critic_hidden_layers = cfg.get("residual_critic_hidden_layers", 2)

    lqr_F_np = lqr.F.numpy().astype(np.float32)

    # Vectorized training envs
    def make_env():
        env = gym.make(cfg["env_name"])
        return ResidualPolicyEnv(env, base_policy, lqr, latent_dim, z_ref_limit)

    vec_env = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    wrapped_env = wrap_env(vec_env)

    # Eval env (vectorized if num_parallel_evals > 1)
    eval_env = make_eval_env(cfg)

    # Models
    obs_space = wrapped_env.observation_space
    act_space = wrapped_env.action_space

    models = {}
    models["policy"] = StochasticActor(obs_space, act_space, device,
                                       hidden_size=actor_hidden_size,
                                       hidden_layers=actor_hidden_layers)
    models["critic_1"] = Critic(obs_space, act_space, device,
                                hidden_size=critic_hidden_size,
                                hidden_layers=critic_hidden_layers)
    models["critic_2"] = Critic(obs_space, act_space, device,
                                hidden_size=critic_hidden_size,
                                hidden_layers=critic_hidden_layers)
    models["target_critic_1"] = Critic(obs_space, act_space, device,
                                       hidden_size=critic_hidden_size,
                                       hidden_layers=critic_hidden_layers)
    models["target_critic_2"] = Critic(obs_space, act_space, device,
                                       hidden_size=critic_hidden_size,
                                       hidden_layers=critic_hidden_layers)

    # Memory
    memory = RandomMemory(memory_size=cfg.get("residual_memory_size", 100000),
                          num_envs=num_envs, device=device)

    # SAC config
    sac_cfg = SAC_DEFAULT_CONFIG.copy()
    sac_cfg["gradient_steps"] = 1
    sac_cfg["batch_size"] = cfg.get("residual_batch_size", 256)
    sac_cfg["discount_factor"] = cfg.get("residual_gamma", 0.99)
    sac_cfg["polyak"] = 1.0 - cfg.get("residual_tau", 0.005)
    sac_cfg["actor_learning_rate"] = cfg.get("residual_lr", 3e-4)
    sac_cfg["critic_learning_rate"] = cfg.get("residual_lr", 3e-4)
    sac_cfg["learn_entropy"] = True
    sac_cfg["initial_entropy_value"] = cfg.get("residual_initial_entropy_value", 1.0)
    sac_cfg["random_timesteps"] = cfg.get("residual_random_timesteps", 1000)
    sac_cfg["learning_starts"] = cfg.get("residual_learning_starts", 1000)
    sac_cfg["experiment"]["write_interval"] = num_envs * 10
    sac_cfg["experiment"]["checkpoint_interval"] = 0
    sac_cfg["experiment"]["directory"] = phase_dir
    sac_cfg["experiment"]["experiment_name"] = cfg.get("run_name","exp_default")
    sac_cfg["experiment"]["store_separately"] = False

    # Agent
    agent = SAC(models=models, memory=memory, cfg=sac_cfg,
                observation_space=obs_space, action_space=act_space, device=device)

    # Training loop
    print(f"\n=== Residual Policy Training (SAC) ===")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Num envs: {num_envs}")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Latent dim (z_ref size): {latent_dim}")
    print(f"  z_ref limit: {z_ref_limit}")
    print(f"  LQR F shape: {lqr_F_np.shape}")
    print(f"  Actor hidden: {actor_hidden_size} x {actor_hidden_layers} layers")
    print(f"  Critic hidden: {critic_hidden_size} x {critic_hidden_layers} layers")

    agent.init()
    writer = agent.writer  # use SKRL's writer for all TB logging

    states, infos = wrapped_env.reset()
    total_steps = 0
    next_eval = 0
    best_reward = -float("inf")
    best_path = os.path.join(phase_dir, "best.pt")

    while total_steps < total_timesteps:
        with torch.no_grad():
            actions = agent.act(states, total_steps, total_timesteps)[0]

        next_states, rewards, terminated, truncated, infos = wrapped_env.step(actions)
        agent.record_transition(states, actions, rewards, next_states,
                                terminated, truncated, infos, total_steps, total_timesteps)
        agent.post_interaction(total_steps, total_timesteps)

        states = next_states
        total_steps += num_envs

        # Periodic evaluation
        if total_steps >= next_eval:
            print(f"\n--- Eval at step {total_steps} ---")
            results = run_eval(models["policy"], base_policy, lqr_F_np,
                               z_ref_limit, cfg, eval_env, writer, total_steps, device)
            avg_reward = results['combined']['reward']['mean']
            print(f"  Success rate: {results['success_rate']:.1%}")
            print(f"  Avg reward: {avg_reward:.1f}")

            # Save best
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(models["policy"].state_dict(), best_path)
                print(f"  New best (reward={best_reward:.1f}) saved to {best_path}")

            # Optionally save every checkpoint
            if keep_all_ckpts:
                ckpt_path = os.path.join(ckpt_dir, f"step_{total_steps}.pt")
                agent.save(ckpt_path)

            next_eval += eval_interval

    # Final eval
    print(f"\n--- Final eval at step {total_steps} ---")
    results = run_eval(models["policy"], base_policy, lqr_F_np,
                       z_ref_limit, cfg, eval_env, writer, total_steps, device)
    avg_reward = results['combined']['reward']['mean']
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Avg reward: {avg_reward:.1f}")

    if avg_reward > best_reward:
        best_reward = avg_reward
        torch.save(models["policy"].state_dict(), best_path)
        print(f"  New best (reward={best_reward:.1f}) saved to {best_path}")

    print(f"Best policy reward: {best_reward:.1f}")

    writer.close()
    eval_env.close()
    vec_env.close()

    print(f"\nResidual training complete. Results in {phase_dir}")
    return models["policy"]


def final_benchmark(residual_model, base_policy, lqr, cfg, run_dir, baseline_stats_path):
    """Run final eval of combined policy and compare against baseline.

    Args:
        residual_model: trained SKRL actor model
        base_policy: callable(obs) -> action
        lqr: LQR object
        cfg: config dict
        run_dir: output directory (final_eval/ created inside)
        baseline_stats_path: path to baseline eval_stats.yaml
    """
    print("\n=== Final Combined Policy Benchmark ===")
    phase_dir = os.path.join(run_dir, "final_eval")
    os.makedirs(phase_dir, exist_ok=True)

    device = next(residual_model.parameters()).device
    lqr_F_np = lqr.F.numpy().astype(np.float32)
    z_ref_limit = cfg.get("residual_z_ref_limit", 1.0)
    residual_model.eval()

    env = make_eval_env(cfg)
    act_space = env.single_action_space if hasattr(env, 'num_envs') else env.action_space
    action_bounds = (act_space.low, act_space.high)
    composite_policy = make_composite_policy(base_policy, residual_model, lqr_F_np, z_ref_limit, device, action_bounds)

    results, all_states, all_actions = evaluate_policy(env, composite_policy, cfg)
    env.close()

    # Save results
    stats_path = os.path.join(phase_dir, "eval_stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"Stats saved to {stats_path}")

    # Load baseline and compare
    baseline_results = load_eval_stats(baseline_stats_path)

    higher_is_better = {"reward", "success_rate"}
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Final':>12} {'Delta':>12}")
    print("-" * 65)

    base_sr = baseline_results["success_rate"]
    final_sr = results["success_rate"]
    delta_sr = final_sr - base_sr
    color = GREEN if delta_sr >= 0 else RED
    print(f"{'success_rate':<25} {base_sr:>11.1%} {final_sr:>11.1%} {color}{delta_sr:>+11.1%}{RESET}")

    metrics = ["length", "energy", "control_torque", "angular_velocity", "reward"]
    for key in metrics:
        base_val = baseline_results["combined"][key]["mean"]
        final_val = results["combined"][key]["mean"]
        delta = final_val - base_val
        if key in higher_is_better:
            color = GREEN if delta >= 0 else RED
        else:
            color = GREEN if delta <= 0 else RED
        print(f"{key:<25} {base_val:>12.3f} {final_val:>12.3f} {color}{delta:>+12.3f}{RESET}")

    print(f"\nFinal benchmark complete. Results in {phase_dir}")
    return results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train residual policy with SAC")
    parser.add_argument("model_dir", type=str,
                        help="Path to Koopman model directory (contains config.yaml, koopman_ckpt.pt, lyapunov_variables.yaml)")
    parser.add_argument("--config", type=str, default="config/residual_debug.yaml",
                        help="Residual training config (learning params, network sizes, run_name)")
    parser.add_argument("--keep-all-ckpts", action="store_true", default=False,
                        help="Save every checkpoint (not just best)")
    parser.add_argument("--base-eval", type=str, default=None,
                        help="Path to baseline eval_stats.yaml for final comparison")
    args = parser.parse_args()

    # 1. Load Koopman config (env, model, policy, LQR params)
    koopman_config_path = os.path.join(args.model_dir, "config.yaml")
    with open(koopman_config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded Koopman config from {koopman_config_path}")

    # 2. Load residual training config and merge
    with open(args.config) as f:
        res_cfg = yaml.safe_load(f)
    cfg.update(res_cfg)
    print(f"Loaded residual config from {args.config}")

    # 3. Load Koopman model
    from model.autoencoder import KoopmanAutoencoder
    from controllers.lqr import LQR
    from launch.eval_policy import make_policy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    koopman_state_dim = cfg["state_dim"]
    if cfg.get("augment_state", False):
        koopman_state_dim += cfg["action_dim"]

    model = KoopmanAutoencoder(
        state_dim=koopman_state_dim,
        latent_dim=cfg["latent_dim"],
        action_dim=cfg["action_dim"],
        k_type=cfg["k_type"],
        encoder_type=cfg.get("encoder_type", "linear"),
        rho=cfg["rho"],
        encoder_spec_norm=cfg.get("encoder_spec_norm", False),
        encoder_latent=cfg.get("encoder_latent", 64),
    ).to(device)

    ckpt_path = os.path.join(args.model_dir, "koopman_ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded Koopman model from {ckpt_path}")

    # 4. Load Lyapunov variables for eta (z_ref_limit)
    lyap_path = os.path.join(args.model_dir, "lyapunov_variables.yaml")
    with open(lyap_path) as f:
        lyap_vars = yaml.safe_load(f)
    z_ref_limit = lyap_vars.get("eta", 1.0)
    print(f"z_ref_limit (eta): {z_ref_limit:.6f}")

    # 5. Build LQR from Koopman model
    A = model.A.detach().cpu()
    B_mat = model.B_matrix.detach().cpu()
    latent_dim = cfg["latent_dim"]
    action_dim = cfg["action_dim"]
    Q = torch.eye(latent_dim) * cfg.get("q_scale", 1.0)
    R = torch.eye(action_dim) * cfg.get("r_scale", 1.0)
    lqr = LQR(A, B_mat, Q, R)
    print(f"LQR: F shape={lqr.F.shape}, gain_norm={lqr.gain_norm:.4f}")

    # 6. Build base policy
    if cfg.get("no_base_policy", False):
        base_policy = None
        print("No base policy — using random actions only")
    else:
        base_policy = make_policy(cfg)

    # 7. Set up output directory
    import shutil
    run_name = cfg.get("run_name", "residual_debug")
    run_dir = os.path.join(args.model_dir, run_name)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
        print(f"Removed existing {run_dir}")
    os.makedirs(run_dir)
    print(f"Output directory: {run_dir}")

    # Save merged config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # 8. Train
    trained_model = train_residual(base_policy, lqr, cfg, run_dir,
                                   z_ref_limit=z_ref_limit,
                                   keep_all_ckpts=args.keep_all_ckpts)

    # 9. Final benchmark against baseline
    if args.base_eval:
        final_benchmark(trained_model, base_policy, lqr, cfg, run_dir, args.base_eval)
    else:
        print("\nSkipping final benchmark (no --base-eval provided)")

    print(f"\n=== Done. All outputs in {run_dir} ===")
