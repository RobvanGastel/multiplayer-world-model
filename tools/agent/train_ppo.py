from __future__ import annotations

import argparse
import functools
import os
import random

import gymnasium as gym

import wm.env.match  # noqa: F401  -- registers AirHockeyMatch-v0 with gymnasium
from wm.policies.ppo import PPO
from wm.utils import merge_config


def log_episode(global_step: int, info: dict):
    print(f"global_step={global_step}  "
          f"episodic_return={info['episode']['r']}  episodic_length={info['episode']['l']}")


def make_eval_callback(eval_dir: str, frame_size: int = 96):
    os.makedirs(eval_dir, exist_ok=True)

    def on_eval(iteration: int, checkpoint_path: str):
        from wm.env.arena import ArenaConfig
        from wm.env.render_match import ROUND_SECONDS, load_policy, policy_trained, play_match
        from wm.env.render import save_gif

        cfg = ArenaConfig(n_mallets=2, max_steps=int(ROUND_SECONDS / ArenaConfig().dt),
                           frame_size=frame_size)
        ppo_eval = load_policy(checkpoint_path, cfg)
        striker = functools.partial(policy_trained, ppo=ppo_eval, deterministic=True)
        defender = functools.partial(policy_trained, ppo=ppo_eval, deterministic=True, mirror=True)
        seed = random.SystemRandom().randint(0, 2**31 - 1)

        frames, final_score = play_match(cfg, seed, px=frame_size, fps=20, hold_secs=0.5,
                                          striker=striker, defender=defender,
                                          win_games=1, best_of=1)
        out_path = os.path.join(eval_dir, f"iter{iteration:06d}_seed{seed}.gif")
        save_gif(frames, out_path, fps=20)
        print(f"eval: iteration={iteration}  wrote {out_path}  final_score={final_score}")

    return on_eval


def train_ppo_agent(args: argparse.Namespace):
    def make_env(env_id: str):
        def thunk():
            env = gym.make(env_id)
            return gym.wrappers.RecordEpisodeStatistics(env)
        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)],
                                     autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    on_eval = None
    if args.eval_every > 0:
        eval_dir = args.eval_dir or (os.path.splitext(args.save_path)[0] + "_eval")
        on_eval = make_eval_callback(eval_dir, frame_size=args.eval_frame_size)

    ppo = PPO(envs, cfg=args)
    if args.load_path:
        ppo.load(args.load_path)
        print(f"resumed weights from {args.load_path}")
    
    ppo.train_loop(on_episode_end=log_episode,
                    checkpoint_path=args.save_path, checkpoint_every=args.checkpoint_every,
                    self_play=args.self_play, opponent_pool_size=args.opponent_pool_size,
                    snapshot_every=args.snapshot_every, min_snapshot_lag=args.min_snapshot_lag,
                    scripted_opponent_prob=args.scripted_opponent_prob,
                    eval_every=args.eval_every, on_eval=on_eval)
    print(f"saved weights to {args.save_path}")
    envs.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/ppo/ppo.yml",
                    help="yaml file of defaults")
    p.add_argument("--env-id", type=str, default="CartPole-v1", help="the id of the environment")
    p.add_argument("--save-path", type=str, default="runs/ppo.pt",
                    help="where to save network weights")
    p.add_argument("--load-path", type=str, default="",
                    help="resume from an existing checkpoint's weights instead of a fresh random")
    raw_args = p.parse_args()
    args = merge_config(raw_args.config, raw_args)
    args.action_logstd_range = tuple(args.action_logstd_range)
    
    train_ppo_agent(args)