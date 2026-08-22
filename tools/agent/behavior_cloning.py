from __future__ import annotations
import argparse

import gymnasium as gym
import numpy as np
import torch

import wm.env.match  # noqa: F401 -- registers AirHockeyMatch-v0
from wm.env.arena import ArenaConfig, ArenaOracle
from wm.env.policies import policy_strike_to_goal, policy_defend_goal
from wm.env.match import ENV_ID, DROP_SPEED_MIN_FRAC, DROP_SPEED_MAX_FRAC, agent_observation
from wm.policies.ppo import PPO


def collect_dataset(cfg: ArenaConfig, n_episodes: int, seed0: int = 0):
    """Rolls out n_episodes scripted-vs-scripted rounds, returns
    (observations, actions) arrays in mallet 0's reference frame."""
    obs_list, act_list = [], []

    for ep in range(n_episodes):
        o = ArenaOracle(cfg, seed=seed0 + ep)
        strike = policy_strike_to_goal(o, 0)
        advance = bool(o.rng.random() < 0.5)
        defend = policy_defend_goal(o, 1, side="right", advance=advance)

        speed = o.rng.uniform(DROP_SPEED_MIN_FRAC, DROP_SPEED_MAX_FRAC) * cfg.puck_max_speed
        o.puck.position = (cfg.width * 0.5, cfg.puck_radius * 1.3)
        o.puck.velocity = (0.0, speed)

        while not o.done:
            a0 = strike()
            a1 = defend()

            obs_list.append(agent_observation(o, mallet_idx=0))
            act_list.append([np.clip(a0[0], -1.0, 1.0), np.clip(a0[1], -1.0, 1.0)])

            obs1_mirrored = agent_observation(o, mallet_idx=1, mirror=True)
            obs_list.append(obs1_mirrored)
            act_list.append([np.clip(-a1[0], -1.0, 1.0), np.clip(a1[1], -1.0, 1.0)])

            o.step([a0, a1])

    return (np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32))


def behavior_clone(ppo: PPO, obs: np.ndarray, act: np.ndarray, epochs: int, batch_size: int, lr: float):
    """Supervised regression of actor_mean onto the scripted actions"""
    device = ppo.device
    obs_t = torch.as_tensor(obs, device=device)
    act_t = torch.as_tensor(act, device=device)
    opt = torch.optim.Adam(ppo.actor_mean.parameters(), lr=lr)

    n = len(obs_t)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = ppo.actor_mean(obs_t[idx])
            loss = ((pred - act_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch}/{epochs}  mse={total / n:.4f}")


def clone_behavior(args):
    cfg = ArenaConfig(n_mallets=2)
    print(f"rolling out {args.episodes} scripted-vs-scripted rounds...")
    obs, act = collect_dataset(cfg, args.episodes, seed0=args.seed)
    print(f"dataset: {len(obs)} (observation, action) examples")

    dummy_envs = gym.vector.SyncVectorEnv([lambda: gym.make(ENV_ID)])
    cfg = argparse.Namespace(seed=args.seed, torch_deterministic=True, cuda=False,
                              hidden_dim=128, learning_rate=2.5e-4,
                              action_mean_clamp=2.0, action_logstd_range=(-2.0, 0.5))
    ppo = PPO(dummy_envs, cfg=cfg)
    dummy_envs.close()

    behavior_clone(ppo, obs, act, args.epochs, args.batch_size, args.lr)

    ppo.save(args.out)
    print(f"saved distilled policy to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400, help="scripted-vs-scripted rounds to roll out")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/distilled.pt")
    args = ap.parse_args()

    clone_behavior(args)
