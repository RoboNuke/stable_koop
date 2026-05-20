"""Residual-policy SAC training loop.

The base controller (LQR in the Koopman latent space) is baked into the
:class:`ResidualPolicyEnv` wrapper, so the SAC actor produces ``z_ref``
directly and no composite policy is needed at training or eval time.

Env stack (gymnasium):
    raw gym env x N -> SyncVectorEnv -> GymVectorAdapter -> ResidualPolicyEnv
The result is a batched-tensor env (IsaacLab-shaped). For Isaac Lab
training, omit the gymnasium stack and pass the Lab env directly to
:class:`ResidualPolicyEnv`.
"""

from __future__ import annotations

import os
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

from models.residual_policy import Critic, StochasticActor
from wrappers.gym_vec_adapter import GymVectorAdapter, TensorToGymAdapter
from wrappers.residual import ResidualPolicyEnv


def _make_residual_env(
    *,
    n_envs: int,
    make_raw_env_fn: Callable,
    koopman_model,
    lqr,
    gamma_max: float,
    gamma_worst_case: float,
    reward_weight: float,
    pred_error_space: str,
    z_ref_max_mode: str,
    obs_augmentation: str,
    disable_action_augmentation: bool,
    device: torch.device,
) -> ResidualPolicyEnv:
    """Build the standard gymnasium -> batched-tensor stack."""
    vec_env = gym.vector.SyncVectorEnv([make_raw_env_fn for _ in range(n_envs)])
    adapter = GymVectorAdapter(vec_env, device=device)
    return ResidualPolicyEnv(
        adapter,
        koopman_model=koopman_model,
        lqr=lqr,
        gamma_max=gamma_max,
        gamma_worst_case=gamma_worst_case,
        reward_weight=reward_weight,
        pred_error_space=pred_error_space,
        z_ref_max_mode=z_ref_max_mode,
        obs_augmentation=obs_augmentation,
        disable_action_augmentation=disable_action_augmentation,
        device=device,
    )


def train_residual(
    *,
    koopman_model,
    lqr,
    gamma_max: float,
    gamma_worst_case: float,
    reward_weight: float,
    pred_error_space: str,
    z_ref_max_mode: str,
    obs_augmentation: str,
    disable_action_augmentation: bool,
    cfg: dict,
    run_dir: str,
    make_env_fn: Callable,
    make_eval_env_fn: Callable,  # kept for signature compatibility; unused
    evaluate_fn: Callable,
    device: torch.device,
    keep_all_ckpts: bool = False,
):
    """SAC training loop. Returns the trained actor model."""
    phase_dir = os.path.join(run_dir, "residual_train")
    os.makedirs(phase_dir, exist_ok=True)
    ckpt_dir = os.path.join(phase_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    num_envs = cfg["num_envs"]
    total_timesteps = cfg["total_timesteps"]
    eval_interval = cfg["eval_interval"]
    actor_hidden_size = cfg["actor_hidden_size"]
    actor_hidden_layers = cfg["actor_hidden_layers"]
    critic_hidden_size = cfg["critic_hidden_size"]
    critic_hidden_layers = cfg["critic_hidden_layers"]

    common_kwargs = dict(
        koopman_model=koopman_model,
        lqr=lqr,
        gamma_max=gamma_max,
        gamma_worst_case=gamma_worst_case,
        reward_weight=reward_weight,
        pred_error_space=pred_error_space,
        z_ref_max_mode=z_ref_max_mode,
        obs_augmentation=obs_augmentation,
        disable_action_augmentation=disable_action_augmentation,
        device=device,
    )

    wrapped_env = _make_residual_env(
        n_envs=num_envs, make_raw_env_fn=make_env_fn, **common_kwargs
    )
    # Eval env: same residual stack, separate instance so eval doesn't
    # disturb training state. Wrap in TensorToGymAdapter so the existing
    # numpy-based evaluator can drive it.
    eval_tensor_env = _make_residual_env(
        n_envs=num_envs, make_raw_env_fn=make_env_fn, **common_kwargs
    )
    eval_env = TensorToGymAdapter(eval_tensor_env)

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
        memory_size=cfg["memory_size"], num_envs=num_envs, device=device
    )

    sac_cfg = SAC_DEFAULT_CONFIG.copy()
    sac_cfg["gradient_steps"] = 1
    sac_cfg["batch_size"] = cfg["batch_size"]
    sac_cfg["discount_factor"] = cfg["gamma"]
    sac_cfg["polyak"] = 1.0 - cfg["tau"]
    sac_cfg["actor_learning_rate"] = cfg["lr"]
    sac_cfg["critic_learning_rate"] = cfg["lr"]
    sac_cfg["learn_entropy"] = True
    sac_cfg["initial_entropy_value"] = cfg["initial_entropy_value"]
    sac_cfg["random_timesteps"] = cfg["random_timesteps"]
    sac_cfg["learning_starts"] = cfg["learning_starts"]
    sac_cfg["experiment"]["write_interval"] = num_envs * 10
    sac_cfg["experiment"]["checkpoint_interval"] = 0
    sac_cfg["experiment"]["directory"] = phase_dir
    sac_cfg["experiment"]["experiment_name"] = cfg["experiment_name"]
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
    print(f"  Action dim: {act_space.shape[0]}")
    print(f"  gamma_max: {gamma_max:.4e}, gamma_worst_case: {gamma_worst_case:.4e}")
    print(f"  reward_weight: {reward_weight}")
    print(f"  pred_error_space: {pred_error_space}")
    print(f"  z_ref_max_mode: {z_ref_max_mode}")
    print(f"  obs_augmentation: {obs_augmentation}")
    print(f"  disable_action_augmentation: {disable_action_augmentation}")

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

        def policy(obs):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                z_ref = residual_model.act({"states": obs_t})[0]
            return z_ref.cpu().numpy().flatten()

        def batch_policy(obs_batch):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_batch, device=device, dtype=torch.float32)
                z_refs = residual_model.act({"states": obs_t})[0]
            return z_refs.cpu().numpy()

        policy.batch = batch_policy
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
    wrapped_env.close()

    print(f"\nResidual training complete. Results in {phase_dir}")
    return models["policy"]
