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
from torch.utils.tensorboard import SummaryWriter

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env

from launch.eval_policy import evaluate as evaluate_policy
from launch.train_pendulum import energy_shaping_policy
from model.residual import StochasticActor, Critic
from wrappers.residual import ResidualPolicyEnv


# ---------------------------------------------------------------------------
# Evaluation callback
# ---------------------------------------------------------------------------

def run_eval(residual_model, base_policy, lqr_F, cfg, eval_env, writer, step, device):
    """Evaluate composite policy and log to TensorBoard.

    Args:
        residual_model: SKRL actor model (outputs z_ref)
        base_policy: callable(obs) -> base_action
        lqr_F: numpy array (action_dim, latent_dim), the LQR gain matrix
        cfg: config dict
        eval_env: raw gymnasium env (no wrapper)
        writer: TensorBoard SummaryWriter
        step: current training step
        device: torch device
    """
    residual_model.eval()

    def composite_policy(obs):
        base_action = base_policy(obs)
        obs_aug = np.concatenate([obs, base_action]).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_aug).unsqueeze(0).to(device)
            z_ref = residual_model.act({"states": obs_t})[0]
            z_ref_np = z_ref.cpu().numpy().flatten()
        u_res = lqr_F @ z_ref_np
        env_low = eval_env.action_space.low
        env_high = eval_env.action_space.high
        return np.clip(base_action + u_res, env_low, env_high)

    results, _, _ = evaluate_policy(eval_env, composite_policy, cfg)

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

def train_residual(base_policy, lqr, cfg, run_dir):
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
    tb_dir = os.path.join(phase_dir, "tensorboard")
    ckpt_dir = os.path.join(phase_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    latent_dim = cfg["latent_dim"]
    z_ref_limit = cfg.get("residual_z_ref_limit", 1.0)
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

    # Single eval env (raw, for evaluate_policy)
    eval_env = gym.make(cfg["env_name"])

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
    sac_cfg["experiment"]["write_interval"] = 0
    sac_cfg["experiment"]["checkpoint_interval"] = 0
    sac_cfg["experiment"]["directory"] = ckpt_dir

    # Agent
    agent = SAC(models=models, memory=memory, cfg=sac_cfg,
                observation_space=obs_space, action_space=act_space, device=device)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=tb_dir)

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

    states, infos = wrapped_env.reset()
    total_steps = 0
    next_eval = 0

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
                               cfg, eval_env, writer, total_steps, device)
            print(f"  Success rate: {results['success_rate']:.1%}")
            print(f"  Avg reward: {results['combined']['reward']['mean']:.1f}")

            agent.save(os.path.join(ckpt_dir, f"step_{total_steps}.pt"))
            next_eval += eval_interval

    # Final eval
    print(f"\n--- Final eval at step {total_steps} ---")
    results = run_eval(models["policy"], base_policy, lqr_F_np,
                       cfg, eval_env, writer, total_steps, device)
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Avg reward: {results['combined']['reward']['mean']:.1f}")

    # Save final policy
    final_path = os.path.join(phase_dir, "final_policy.pt")
    torch.save(models["policy"].state_dict(), final_path)
    print(f"Final policy saved to {final_path}")

    writer.close()
    eval_env.close()
    vec_env.close()

    print(f"\nResidual training complete. Results in {phase_dir}")
    return models["policy"]


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train residual policy with SAC")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from controllers.lqr import LQR

    # Build a dummy LQR for standalone testing (normally comes from Koopman model)
    latent_dim = cfg["latent_dim"]
    action_dim = cfg["action_dim"]
    A = torch.eye(latent_dim) * 0.9
    B = torch.randn(latent_dim, action_dim) * 0.1
    Q = torch.eye(latent_dim) * cfg.get("q_scale", 1.0)
    R = torch.eye(action_dim) * cfg.get("r_scale", 1.0)
    lqr = LQR(A, B, Q, R)
    print(f"Standalone LQR: F shape={lqr.F.shape}, gain_norm={lqr.gain_norm:.4f}")

    base_policy = lambda obs: energy_shaping_policy(obs)
    run_dir = cfg.get("residual_run_dir", "output/residual_standalone")
    os.makedirs(run_dir, exist_ok=True)

    train_residual(base_policy, lqr, cfg, run_dir)
