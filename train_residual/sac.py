"""Residual-policy SAC training core (env-loop and composite policy builder).

Ported from ``launch/train_residual.py``. Only changes: imports re-pointed
at the refactored modules, and the SAC loop now takes a pre-built ``cfg``
dict (legacy flat shape) so the training loop stays bit-equivalent.
"""

from __future__ import annotations

import os
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import yaml

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory

from models.residual_policy import Critic, StochasticActor
from wrappers.residual import ResidualPolicyEnv


def make_composite_policy(base_policy, residual_model, lqr_F_np, z_ref_limit, device, action_bounds):
    """Build a callable that combines a base policy with an LQR residual derived from z_ref.

    Exposes both single-obs and batched call paths (``policy(obs)`` and
    ``policy.batch(obs_batch)``).
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
        N = len(obs_batch)
        base_actions = np.array([base_policy(obs_batch[i]) for i in range(N)])
        obs_aug = np.concatenate([obs_batch, base_actions], axis=-1).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_aug).to(device)
            raw_actions = residual_model.act({"states": obs_t})[0]
            raw_actions_np = raw_actions.cpu().numpy()
        z_refs = z_ref_limit * raw_actions_np
        u_res = z_refs @ lqr_F_np.T
        total = base_actions + u_res
        return np.clip(total, act_low, act_high)

    policy.batch = batch_policy
    return policy


def train_residual(
    base_policy: Callable,
    lqr,
    cfg: dict,
    run_dir: str,
    *,
    make_env_fn: Callable,
    make_eval_env_fn: Callable,
    evaluate_fn: Callable,
    z_ref_limit: float = 1.0,
    keep_all_ckpts: bool = False,
):
    """SAC training loop. Returns the trained actor model.

    Caller injects env builders and the evaluation callback to keep this
    module independent of ``eval/``.
    """
    phase_dir = os.path.join(run_dir, "residual_train")
    os.makedirs(phase_dir, exist_ok=True)
    ckpt_dir = os.path.join(phase_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    latent_dim = cfg["latent_dim"]
    num_envs = cfg.get("residual_num_envs", 8)
    total_timesteps = cfg.get("residual_total_timesteps", 100000)
    eval_interval = cfg.get("residual_eval_interval", 5000)
    actor_hidden_size = cfg.get("residual_actor_hidden_size", 64)
    actor_hidden_layers = cfg.get("residual_actor_hidden_layers", 2)
    critic_hidden_size = cfg.get("residual_critic_hidden_size", 64)
    critic_hidden_layers = cfg.get("residual_critic_hidden_layers", 2)
    lqr_F_np = lqr.F.numpy().astype(np.float32)

    def make_env():
        env = make_env_fn()
        return ResidualPolicyEnv(env, base_policy, lqr, latent_dim, z_ref_limit)

    vec_env = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])
    wrapped_env = wrap_env(vec_env)
    eval_env = make_eval_env_fn()

    obs_space = wrapped_env.observation_space
    act_space = wrapped_env.action_space

    models = {
        "policy": StochasticActor(obs_space, act_space, device, hidden_size=actor_hidden_size, hidden_layers=actor_hidden_layers),
        "critic_1": Critic(obs_space, act_space, device, hidden_size=critic_hidden_size, hidden_layers=critic_hidden_layers),
        "critic_2": Critic(obs_space, act_space, device, hidden_size=critic_hidden_size, hidden_layers=critic_hidden_layers),
        "target_critic_1": Critic(obs_space, act_space, device, hidden_size=critic_hidden_size, hidden_layers=critic_hidden_layers),
        "target_critic_2": Critic(obs_space, act_space, device, hidden_size=critic_hidden_size, hidden_layers=critic_hidden_layers),
    }

    memory = RandomMemory(
        memory_size=cfg.get("residual_memory_size", 100000), num_envs=num_envs, device=device
    )

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
    sac_cfg["experiment"]["experiment_name"] = cfg.get("experiment_name", "residual")
    sac_cfg["experiment"]["store_separately"] = False

    agent = SAC(
        models=models,
        memory=memory,
        cfg=sac_cfg,
        observation_space=obs_space,
        action_space=act_space,
        device=device,
    )

    print("\n=== Residual Policy Training (SAC) ===")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Num envs: {num_envs}")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Latent dim (z_ref size): {latent_dim}")
    print(f"  z_ref limit: {z_ref_limit}")
    print(f"  LQR F shape: {lqr_F_np.shape}")

    agent.init()
    writer = agent.writer

    states, infos = wrapped_env.reset()
    total_steps = 0
    next_eval = 0
    best_reward = -float("inf")
    best_path = os.path.join(phase_dir, "best.pt")
    print("(Press Ctrl+C to end training early and continue pipeline)")

    def _run_eval(step):
        residual_model = models["policy"]
        residual_model.eval()
        act_space_local = (
            eval_env.single_action_space if hasattr(eval_env, "num_envs") else eval_env.action_space
        )
        action_bounds = (act_space_local.low, act_space_local.high)
        policy = make_composite_policy(
            base_policy, residual_model, lqr_F_np, z_ref_limit, device, action_bounds
        )
        results, _, _ = evaluate_fn(eval_env, policy, cfg)
        writer.add_scalar("eval_total_metric/success_rate", results["success_rate"], step)
        residual_model.train()
        return results

    try:
        while total_steps < total_timesteps:
            with torch.no_grad():
                actions = agent.act(states, total_steps, total_timesteps)[0]
            next_states, rewards, terminated, truncated, infos = wrapped_env.step(actions)
            agent.record_transition(
                states, actions, rewards, next_states, terminated, truncated, infos, total_steps, total_timesteps
            )
            agent.post_interaction(total_steps, total_timesteps)
            states = next_states
            total_steps += num_envs

            if total_steps >= next_eval:
                print(f"\n--- Eval at step {total_steps} ---")
                results = _run_eval(total_steps)
                avg_reward = results["combined"]["reward"]["mean"]
                print(f"  Success rate: {results['success_rate']:.1%}")
                print(f"  Avg reward: {avg_reward:.1f}")
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    torch.save(models["policy"].state_dict(), best_path)
                    print(f"  New best (reward={best_reward:.1f}) saved to {best_path}")
                if keep_all_ckpts:
                    ckpt_path = os.path.join(ckpt_dir, f"step_{total_steps}.pt")
                    agent.save(ckpt_path)
                next_eval += eval_interval
    except KeyboardInterrupt:
        print(f"\nResidual training interrupted at step {total_steps}. Continuing...")

    print(f"\n--- Final eval at step {total_steps} ---")
    results = _run_eval(total_steps)
    avg_reward = results["combined"]["reward"]["mean"]
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
