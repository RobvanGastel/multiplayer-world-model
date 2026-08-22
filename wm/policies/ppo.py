"""PPO (Schulman et al., 2017), adapted from CleanRL's single-file ppo.py:
https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
"""
from __future__ import annotations

import os
import random
import time
from collections import deque
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO(nn.Module):
    def __init__(self, envs: gym.vector.VectorEnv, cfg, seed_rngs: bool = True):
        super().__init__()
        space = envs.single_action_space
        assert isinstance(space, (gym.spaces.Discrete, gym.spaces.Box)), \
            "only discrete or continuous (Box) action spaces are supported"
        self.continuous = isinstance(space, gym.spaces.Box)
        self.envs = envs
        self.cfg = cfg

        # seed_rngs=False builds a network (e.g. a self-play opponent snapshot
        # holder) without reseeding.
        if seed_rngs:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            torch.backends.cudnn.deterministic = cfg.torch_deterministic
        self.device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        h = cfg.hidden_dim
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, h)), nn.Tanh(),
            _layer_init(nn.Linear(h, h)), nn.Tanh(),
            _layer_init(nn.Linear(h, 1), std=1.0),
        )
        if self.continuous:
            action_dim = int(np.prod(space.shape))
            self.actor_mean = nn.Sequential(
                _layer_init(nn.Linear(obs_dim, h)), nn.Tanh(),
                _layer_init(nn.Linear(h, h)), nn.Tanh(),
                _layer_init(nn.Linear(h, action_dim), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor = nn.Sequential(
                _layer_init(nn.Linear(obs_dim, h)), nn.Tanh(),
                _layer_init(nn.Linear(h, h)), nn.Tanh(),
                _layer_init(nn.Linear(h, space.n), std=0.01),
            )
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=cfg.learning_rate, eps=1e-5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if self.continuous:
            clamp = self.cfg.action_mean_clamp
            action_mean = torch.clamp(self.actor_mean(x), -clamp, clamp)
            logstd_lo, logstd_hi = self.cfg.action_logstd_range
            raw_logstd = self.actor_logstd.expand_as(action_mean)
            action_logstd = logstd_lo + 0.5 * (logstd_hi - logstd_lo) * (torch.tanh(raw_logstd) + 1.0)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def save(self, path: str):
        """Save network weights (not optimizer state) to `path`."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Single-observation inference (no grad, no batch dim in/out)"""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if not deterministic:
                action, _, _, _ = self.get_action_and_value(x)
            elif self.continuous:
                clamp = self.cfg.action_mean_clamp
                action = torch.clamp(self.actor_mean(x), -clamp, clamp)
            else:
                action = self.actor(x).argmax(dim=-1)
        return action.squeeze(0).cpu().numpy()

    def _apply_opponent(self, fn):
        for e in self.envs.envs:
            unwrapped = e.unwrapped
            if hasattr(unwrapped, "opponent_policy"):
                unwrapped.opponent_policy = fn

    def _self_play_opponent(self, pool, frozen, iteration: int, min_snapshot_lag: int,
                             scripted_opponent_prob: float = 0.0):
        if scripted_opponent_prob and random.random() < scripted_opponent_prob:
            return None
        eligible = [sd for (snap_iter, sd) in pool if iteration - snap_iter >= min_snapshot_lag]
        if not eligible:
            return None
        frozen.load_state_dict(random.choice(eligible))
        policy = frozen

        def act(obs):
            return policy.act(obs, deterministic=False)
        return act

    def train_loop(
        self, on_episode_end: Optional[Callable[[int, dict], None]] = None,
        checkpoint_path: Optional[str] = None, checkpoint_every: int = 10,
        self_play: bool = False, opponent_pool_size: int = 8,
        snapshot_every: int = 10, min_snapshot_lag: int = 100,
        scripted_opponent_prob: float = 0.0, eval_every: Optional[int] = None,
        on_eval: Optional[Callable[[int, str], None]] = None,
        best_window: int = 200
    ):
        cfg, envs, device = self.cfg, self.envs, self.device
        batch_size = cfg.num_envs * cfg.num_steps
        minibatch_size = batch_size // cfg.num_minibatches
        num_iterations = cfg.total_timesteps // batch_size

        # best-model tracking
        recent_returns: deque = deque(maxlen=best_window)
        best_win_rate = -1.0
        best_path = None
        if checkpoint_path:
            root, ext = os.path.splitext(checkpoint_path)
            best_path = f"{root}_best{ext}"

        opponent_pool: list = []
        frozen: Optional["PPO"] = None
        if self_play:
            frozen = PPO(envs, cfg=cfg, seed_rngs=False)
            self._apply_opponent(self._self_play_opponent(
                opponent_pool, frozen, 0, min_snapshot_lag, scripted_opponent_prob))

        obs = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((cfg.num_steps, cfg.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
        values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=cfg.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(cfg.num_envs).to(device)

        for iteration in range(1, num_iterations + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            for step in range(cfg.num_steps):
                global_step += cfg.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = self.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.Tensor(next_done).to(device)

                # gymnasium's vector envs merge per-env infos into a dict of
                # arrays (with a "_<key>" boolean presence mask) rather than a
                # list of per-env dicts, so "episode" has to be read per-index.
                final_info = infos.get("final_info")
                if final_info and "episode" in final_info:
                    ep, done_mask = final_info["episode"], final_info["_episode"]
                    for i in range(cfg.num_envs):
                        if done_mask[i]:
                            r = float(ep["r"][i])
                            if on_episode_end:
                                on_episode_end(global_step, {"episode": {"r": r, "l": int(ep["l"][i])}})
                            if best_path:
                                recent_returns.append(r)
                                if len(recent_returns) == best_window:
                                    win_rate = sum(1 for x in recent_returns if x == 1.0) / best_window
                                    if win_rate > best_win_rate:
                                        best_win_rate = win_rate
                                        self.save(best_path)
                                        print(f"new best win_rate={win_rate:.3f} (of last {best_window} "
                                              f"episodes) at step={global_step} -> saved {best_path}")

            # bootstrap value if not done, then GAE
            with torch.no_grad():
                next_value = self.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(cfg.num_steps)):
                    if t == cfg.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + cfg.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(batch_size)
            clipfracs = []
            for epoch in range(cfg.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = b_inds[start:start + minibatch_size]
                    mb_actions = b_actions[mb_inds] if self.continuous else b_actions.long()[mb_inds]

                    _, newlogprob, entropy, newvalue = self.get_action_and_value(
                        b_obs[mb_inds], mb_actions)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # http://joschu.net/blog/kl-approx.html
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if cfg.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if cfg.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds], -cfg.clip_coef, cfg.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    if self.continuous:
                        # TODO: Remove?
                        # penalize the RAW mean, not the clamped one -- torch.clamp
                        # has zero gradient outside its range, so penalizing the
                        # clamped value gives no pull-back once the mean's already
                        # past the boundary, which is exactly the case this exists
                        # to fix.
                        mean_penalty = self.actor_mean(b_obs[mb_inds]).pow(2).mean()
                        logstd_penalty = self.actor_logstd.pow(2).mean()
                    else:
                        mean_penalty = torch.zeros((), device=device)
                        logstd_penalty = torch.zeros((), device=device)
                    loss = (pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef
                            + cfg.mean_reg_coef * mean_penalty + cfg.logstd_reg_coef * logstd_penalty)

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            sps = int(global_step / (time.time() - start_time))
            print(f"iteration {iteration}/{num_iterations}  step={global_step}  SPS={sps}  "
                  f"loss/policy={pg_loss.item():.4f}  loss/value={v_loss.item():.4f}  "
                  f"loss/entropy={entropy_loss.item():.4f}  loss/mean_mag={mean_penalty.item():.4f}  "
                  f"approx_kl={approx_kl.item():.4f}  clipfrac={np.mean(clipfracs):.3f}  "
                  f"explained_var={explained_var:.3f}  lr={self.optimizer.param_groups[0]['lr']:.2e}")

            if self_play:
                if iteration % snapshot_every == 0:
                    snapshot = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                    opponent_pool.append((iteration, snapshot))
                    if len(opponent_pool) > opponent_pool_size:
                        opponent_pool.pop(0)
                self._apply_opponent(self._self_play_opponent(
                opponent_pool, frozen, iteration, min_snapshot_lag, scripted_opponent_prob))

            if checkpoint_path and iteration % checkpoint_every == 0:
                self.save(checkpoint_path)

            if eval_every and on_eval and checkpoint_path and iteration % eval_every == 0:
                self.save(checkpoint_path)
                on_eval(iteration, checkpoint_path)

        if checkpoint_path:
            self.save(checkpoint_path)
