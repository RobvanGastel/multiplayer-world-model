from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from wm.env.arena import ArenaConfig, ArenaOracle
from wm.env.policies import policy_defend_goal, make_defend_variant, DEFEND_VARIANTS
from wm.env.render_iso import render_iso

ENV_ID = "AirHockeyMatch-v0"


def agent_observation(oracle: ArenaOracle, mallet_idx: int = 0, mirror: bool = False) -> np.ndarray:
    """The layout of the obseravation space is as follows: The environment as
    defined here observes the underlying game state, a low dimensional vector
    for one mallet: its own absolute [x,y,vx,vy,ax,ay], then the puck and every
    other mallet"""

    s = oracle.state()
    cfg = oracle.cfg
    w, h = cfg.width, cfg.height

    own_x, own_y, own_vx, own_vy = s["mallets"][mallet_idx]
    own_ax, own_ay = s["mallet_accels"][mallet_idx]
    mallet_accel_scale = cfg.thrust / cfg.mallet_mass
    puck_accel_scale = cfg.thrust / cfg.puck_mass

    # Option for relative or absolute coordinates.
    def own_body():
        nx, ny = (own_x - w / 2) / (w / 2), (own_y - h / 2) / (h / 2)
        nvx, nvy = own_vx / cfg.mallet_max_speed, own_vy / cfg.mallet_max_speed
        nax = np.clip(own_ax / mallet_accel_scale, -5.0, 5.0)
        nay = np.clip(own_ay / mallet_accel_scale, -5.0, 5.0)
        if mirror:
            nx, nvx, nax = -nx, -nvx, -nax
        return [nx, ny, nvx, nvy, nax, nay]

    def relative_body(v, accel, other_max_speed, other_accel_scale):
        x, y, vx, vy = v
        ax, ay = accel
        # normalized by the FULL board dimension, not half -- own_x/own_y can
        # be anywhere in [0, w]/[0, h], so the max possible separation is the
        # full w/h, not w/2. Dividing by w/2 (as own_body's own nx/ny do,
        # correctly, since THEIR max excursion from center really is w/2)
        # let dx/dy reach +-2 instead of +-1, unlike every other feature in
        # this observation (velocities are already bounded to [-1,1] via
        # speed_sum's triangle-inequality bound; accelerations are clipped).
        # A relative-position spike outside the range everything else lives
        # in is exactly the kind of out-of-distribution input a network
        # generalizes worst on -- suspected of causing exactly the erratic
        # behavior seen when the puck sits deep behind the mallet, since
        # that's when this separation is largest.
        dx, dy = (x - own_x) / w, (y - own_y) / h
        speed_sum = other_max_speed + cfg.mallet_max_speed
        dvx, dvy = (vx - own_vx) / speed_sum, (vy - own_vy) / speed_sum
        accel_sum = other_accel_scale + mallet_accel_scale
        dax = np.clip((ax - own_ax) / accel_sum, -5.0, 5.0)
        day = np.clip((ay - own_ay) / accel_sum, -5.0, 5.0)
        if mirror:
            dx, dvx, dax = -dx, -dvx, -dax
        return [dx, dy, dvx, dvy, dax, day]

    own = own_body()
    puck = relative_body(s["puck"], s["puck_accel"], cfg.puck_max_speed, puck_accel_scale)
    others = [c for i, m in enumerate(s["mallets"]) if i != mallet_idx
              for c in relative_body(m, s["mallet_accels"][i], cfg.mallet_max_speed, mallet_accel_scale)]
    return np.array(own + puck + others, dtype=np.float32)

# fraction of cfg.puck_max_speed the round's centerline drop can start at.
DROP_SPEED_MIN_FRAC = 0.15
DROP_SPEED_MAX_FRAC = 0.75


class AirHockeyMatchEnv(gym.Env):
    """One striker-vs-defender round per episode."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, cfg: ArenaConfig | None = None, include_pixels: bool = False,
                 render_mode: str | None = None, opponent_policy=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.cfg = cfg or ArenaConfig(n_mallets=2)
        self.include_pixels = include_pixels
        self.render_mode = render_mode
        # (obs: np.ndarray) -> action, or None for the scripted defender. A
        # plain public attribute -- a self-play trainer swaps it in directly
        # (env.unwrapped.opponent_policy = fn) between rollout phases; reset()
        self.opponent_policy = opponent_policy

        # observation: own mallet + puck + every other mallet, each
        # [x, y, vx, vy, ax, ay]. the hidden force field is deliberately excluded
        # -- it's latent state a world model should learn to infer, not
        # something handed to the agent.
        state_dim = 6 * (2 + max(0, self.cfg.n_mallets - 1))
        state_box = spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)
        if include_pixels:
            self.observation_space = spaces.Dict({
                "state": state_box,
                "flat": spaces.Box(0, 255, shape=(self.cfg.frame_size, self.cfg.frame_size, 3),
                                    dtype=np.uint8),
            })
        else:
            self.observation_space = state_box
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.oracle: ArenaOracle | None = None
        self._defend = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        ep_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self.oracle = ArenaOracle(self.cfg, seed=ep_seed)

        # Both possible defenders are built now and self._defend re-checks
        # opponent_policy on every call.
        if self.cfg.n_mallets != 2:
            self._defend = None
        else:
            # sample a defensive style each round from DEFEND_VARIANTS
            # like noise, aggression, advance, defense_only.
            variant = self.oracle.rng.choice(list(DEFEND_VARIANTS.keys()))
            self._scripted_defend = make_defend_variant(
                self.oracle, idx=1, side="right", variant=variant)
            self._network_defend = self._make_opponent_actor()
            self._defend = self._dispatch_defend

        # the round's drop
        cfg = self.cfg
        speed = self.oracle.rng.uniform(DROP_SPEED_MIN_FRAC, DROP_SPEED_MAX_FRAC) * cfg.puck_max_speed
        self.oracle.puck.position = (cfg.width * 0.5, cfg.puck_radius * 1.3)
        self.oracle.puck.velocity = (0.0, speed)

        return self._obs(), {"outcome": None, "state": self.oracle.state()}

    def step(self, action):
        ax = float(np.clip(action[0], -1.0, 1.0))
        ay = float(np.clip(action[1], -1.0, 1.0))
        defend_action = self._defend() if self._defend is not None else None
        actions = [(ax, ay)] + ([defend_action] if defend_action is not None else [])
        state, _ = self.oracle.step(actions)

        outcome = self.oracle.outcome
        # timeout is -1 too, not 0 -- a free draw made "stall until the shot
        # clock runs out" a rational way to lock in a better-than-attacking
        # outcome once a round dragged on, and trained agents learned
        # exactly that (verified: mallet speed drops from ~218 to ~9 over a
        # timeout round's last 20%, parking motionless a reachable ~70 units
        # from the puck). Scoring is now the only non-losing outcome.
        # oracle.outcome is None (not "timeout") on every non-terminal step,
        # so it must stay out of this dict -- .get()'s default has to be 0.0,
        # not -1.0, or every ongoing step gets penalized too.
        reward = {"goal": 1.0, "conceded": -1.0, "timeout": -1.0}.get(outcome, 0.0)
        terminated = outcome in ("goal", "conceded")
        truncated = outcome == "timeout"
        # defend_action is in world/oracle frame (mallet-1's raw coordinates, as fed to
        # oracle.step). Mirror the x-axis to match p1's own rendered viewpoint instead
        # (render_iso's agent_idx=1 camera flip, same convention agent_observation's
        # mirror=True uses for p1's observations) -- so a recorded action reads with the
        # same left/right sense as p0's does relative to p0's own frame, regardless of
        # whether _defend is the network or scripted path (both return world-frame here).
        defender_action = (-defend_action[0], defend_action[1]) if defend_action is not None else None
        return self._obs(), reward, terminated, truncated, {
            "outcome": outcome, "state": state, "defender_action": defender_action,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self.oracle.render_flat()
        return None

    def _make_opponent_actor(self):
        def act():
            obs = agent_observation(self.oracle, mallet_idx=1, mirror=True)
            ax, ay = self.opponent_policy(obs)
            return (-float(ax), float(ay))
        return act

    def _dispatch_defend(self):
        """Reads opponent_policy fresh on every call (not just at reset) so a
        self-play trainer can swap it mid-round."""
        if self.opponent_policy is not None:
            return self._network_defend()
        return self._scripted_defend()

    def _obs(self):
        vec = agent_observation(self.oracle, mallet_idx=0)
        if self.include_pixels:
            return {"state": vec, "flat": self.oracle.render_flat()}
        return vec


class MatchDataRecorder(gym.Wrapper):
    """Records one view (flat or iso, picked via `view`) from both players'
    perspectives, plus low-dim state and actions, for the episode currently
    in progress."""

    def __init__(self, env: AirHockeyMatchEnv, view: str = "flat"):
        super().__init__(env)
        assert view in ("flat", "iso")
        self.view = view
        self._p0, self._p1, self._states, self._actions, self._actions_p1 = [], [], [], [], []

    def _frame(self, oracle, agent_idx: int):
        if self.view == "flat":
            # render_flat has no inherent viewpoint; P1's copy is the board
            # mirrored so their goal reads near, same convention as
            # wm/env/render_match.py's compose_frame.
            flat = oracle.render_flat()
            return flat if agent_idx == 0 else np.fliplr(flat).copy()
        return render_iso(oracle, agent_idx=agent_idx)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        oracle = self.env.unwrapped.oracle
        self._p0 = [self._frame(oracle, 0)]
        self._p1 = [self._frame(oracle, 1)]
        self._states = [info["state"]]
        self._actions = []
        self._actions_p1 = []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        oracle = self.env.unwrapped.oracle
        self._p0.append(self._frame(oracle, 0))
        self._p1.append(self._frame(oracle, 1))
        self._states.append(info["state"])
        self._actions.append(np.asarray(action, dtype=np.float32))
        # defender_action is None only when n_mallets != 2 (see AirHockeyMatchEnv.step) --
        # not a real case for MatchDataRecorder's own 2-mallet usage, but padded with a
        # no-op rather than crashing so this stays usable standalone.
        defender_action = info.get("defender_action")
        self._actions_p1.append(
            np.asarray(defender_action if defender_action is not None else (0.0, 0.0),
                       dtype=np.float32)
        )
        return obs, reward, terminated, truncated, info

    def episode_data(self) -> dict:
        """Everything recorded for the most recently reset/stepped episode."""
        return {
            "view": self.view,
            "frames_p0": np.stack(self._p0).astype(np.uint8),
            "frames_p1": np.stack(self._p1).astype(np.uint8),
            "states": self._states,
            "actions": np.array(self._actions, dtype=np.float32),
            "actions_p1": np.array(self._actions_p1, dtype=np.float32),
            "field": list(self.env.unwrapped.oracle.field),
            "outcome": self.env.unwrapped.oracle.outcome,
        }


# registered on import so `gym.make(ENV_ID)` works
gym.register(id=ENV_ID, entry_point="wm.env.match:AirHockeyMatchEnv")
