from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from wm.data.actions import action_to_wasd, actions_to_wasd
from wm.env.arena import ArenaConfig
from wm.env.match import AirHockeyMatchEnv, MatchDataRecorder
from wm.env.render_iso import BACKGROUND_COLOR, render_iso
from wm.env.render_match import (
    countdown_frames_raw, draw_ramp_iso, drop_preroll_frames, load_policy, panel_frame,
)
from wm.utils import merge_config


def discover_checkpoints(patterns: list[str]) -> list[str]:
    """Union-glob every pattern in `patterns`, de-duplicated, sorted. A
    literal existing path (no wildcard) works as its own one-match
    "pattern" via glob.glob, so an explicit pool -- e.g. --checkpoints-glob
    runs/distilled.pt runs/some_best.pt -- works the same way a wildcard
    does, without needing shell-style {a,b,c} brace expansion (glob.glob
    doesn't support that)."""
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"no checkpoints matched any of {patterns!r}")
    return paths


def _with_ramp_iso(frame: np.ndarray, cfg: ArenaConfig, agent_idx: int, puck_y) -> np.ndarray:
    """Composite the ramp/chute overlay onto an already-rendered iso frame --
    same draw_ramp_iso call wm/env/render_match.py's compose_frame makes for
    every frame of the viewer GIF, not just the intro (the chute shrinks to
    nothing once the puck's a puck-radius or so into the field, per
    _ramp_polygon_world, so this is a no-op visually for most of a game)."""
    im = Image.fromarray(frame, "RGB")
    dd = ImageDraw.Draw(im, "RGBA")
    draw_ramp_iso(dd, cfg, agent_idx, cfg.frame_size, puck_y)
    return np.array(im, dtype=np.uint8)


def intro_frames(oracle, cfg: ArenaConfig,
                  countdown: bool = True) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Ramp-drop preroll + "3, 2, 1, GO!" countdown for one game's start,
    rendered from both players' iso views -- reuses the exact primitives
    wm/env/render_match.py's play_round()/play_match() use for the
    standalone match-viewer GIF (drop_preroll_frames, draw_ramp_iso,
    countdown_frames_raw) so a collected clip's intro is pixel-for-pixel
    the same sequence. Reads the drop speed/position AirHockeyMatchEnv.
    reset() already rolled onto `oracle`, rather than rerolling a new one.
    `countdown=False` (for every game but the series opener, see play_match's
    game_no==1 check, matching wm/env/render_match.py's play_match) skips
    the countdown hold and keeps just the ramp preroll, so a multi-game
    match clip isn't mostly countdown. Returns (p0_frames, p1_frames); pair
    with a no-op action per frame when prepending to a real episode's
    actions -- these aren't real env transitions, the puck's position is
    faked frame-by-frame here exactly like drop_preroll_frames does for the
    viewer."""
    speed = oracle.puck.velocity.y  # vx is always 0 at reset (AirHockeyMatchEnv.reset)

    def snap():
        puck_y = oracle.puck.position.y
        return (_with_ramp_iso(render_iso(oracle, agent_idx=0), cfg, 0, puck_y),
                _with_ramp_iso(render_iso(oracle, agent_idx=1), cfg, 1, puck_y))

    ramp = drop_preroll_frames(oracle, snap, speed)
    p0_ramp, p1_ramp = (list(v) for v in zip(*ramp))

    if not countdown:
        return p0_ramp, p1_ramp

    fps = 1.0 / cfg.dt
    p0_count = countdown_frames_raw(p0_ramp[0], cfg, fps)
    p1_count = countdown_frames_raw(p1_ramp[0], cfg, fps)
    return p0_count + p0_ramp, p1_count + p1_ramp


def stamp_frames(frames: np.ndarray, cfg: ArenaConfig, score_before: tuple[int, int],
                  score_after: tuple[int, int], outcome: str, pad_height: int,
                  hold_frames: int) -> np.ndarray:
    """Burn the running score (this player's score first) into every frame's
    header bar, via wm/env/render_match.py's panel_frame -- the exact same
    per-view drawing compose_frame's 2x2 grid uses, so this isn't a
    reimplementation. Every real frame gets `score_before` (the score
    entering this game) with no overlay, same as the viewer's main play
    loop; then, matching play_match()'s own post-round hold, the final
    frame is repeated `hold_frames` times with `score_after` (post-outcome)
    and the outcome banner -- without this hold there's only one frame to
    read the banner off of and the score never visibly changes. Adds
    header_h rows above each frame (same as the display GIF), then
    bottom-pads (with render_iso's BACKGROUND_COLOR, so the pad reads as part of the board) up to `pad_height` --
    the codec's encoder/decoder
    round-trip needs height to be a multiple of its 32x spatial downsampling
    (see configs/codec/rae_encoder.yml's video.height comment), which the
    raw board+header height usually isn't. Pre-baking this here matches
    what VideoCodec.preprocess_batch would otherwise pad at runtime, so it
    becomes a no-op there for newly-collected clips.
    """
    before_text = f"{score_before[0]}-{score_before[1]}"
    after_text = f"{score_after[0]}-{score_after[1]}"
    banner = {"goal": "P0 SCORES", "conceded": "P1 SCORES"}.get(outcome, "TIME'S UP - DRAW")

    body = np.stack([
        np.array(panel_frame(frames[i], cfg, before_text), dtype=np.uint8)
        for i in range(frames.shape[0])
    ])
    hold_frame = np.array(panel_frame(frames[-1], cfg, after_text, overlay_text=banner),
                           dtype=np.uint8)
    stamped = np.concatenate([body, np.stack([hold_frame] * hold_frames)], axis=0)

    bottom_pad = pad_height - stamped.shape[1]
    if bottom_pad > 0:
        pad_rows = np.full((stamped.shape[0], bottom_pad, stamped.shape[2], 3),
                            BACKGROUND_COLOR, dtype=np.uint8)
        stamped = np.concatenate([stamped, pad_rows], axis=1)
    return stamped


def wasd_to_action(wasd) -> np.ndarray:
    """(w, a, s, d) key presses -> the full-thrust continuous action they stand for (inverse of
    wm.data.actions.action_to_wasd's convention)."""
    w, a, s, d = wasd
    return np.array([float(d) - float(a), float(s) - float(w)], dtype=np.float32)


class ActionShaper:
    """Wraps a policy for data collection, to make the recorded actions informative for the world
    model (which only sees W/A/S/D, and otherwise can read the PPO agent's intent off the frames):

    * sticky random moves: with `random_frac` of steps on average, hold one of the 9 WASD
      directions (incl. no key) for `hold` frames drawn from [hold_min, hold_max], so actions
      carry information the context can't predict;
    * `wasd_only`: execute every action as the WASD keys it is recorded as (see wasd_to_action),
      so the recorded key presses describe the real control exactly instead of a thresholded
      version of the continuous thrust.
    """

    DIRECTIONS = [(w, a, s, d) for w, s in ((0, 0), (1, 0), (0, 1)) for a, d in ((0, 0), (1, 0), (0, 1))]

    def __init__(self, policy, rng: np.random.Generator, random_frac: float = 0.0,
                 hold_min: int = 6, hold_max: int = 30, wasd_only: bool = False):
        self.policy, self.rng, self.wasd_only = policy, rng, wasd_only
        self.hold_min, self.hold_max = hold_min, hold_max
        # Start a random segment with this per-step probability so that on average random_frac of
        # all steps are random: frac = p*L / (1 + p*L) for mean hold L.
        mean_hold = (hold_min + hold_max) / 2
        self.p_start = random_frac / ((1 - random_frac) * mean_hold) if random_frac > 0 else 0.0
        self.remaining = 0
        self.current = None

    def __call__(self, obs) -> np.ndarray:
        if self.remaining == 0 and self.rng.random() < self.p_start:
            self.remaining = int(self.rng.integers(self.hold_min, self.hold_max + 1))
            self.current = wasd_to_action(self.DIRECTIONS[self.rng.integers(len(self.DIRECTIONS))])
        if self.remaining > 0:
            self.remaining -= 1
            return self.current
        action = np.asarray(self.policy(obs), dtype=np.float32)
        return wasd_to_action(action_to_wasd(*action)) if self.wasd_only else action


def play_game(env: MatchDataRecorder, act_p0, seed: int, cfg: ArenaConfig,
              countdown: bool = True) -> dict:
    obs, _ = env.reset(seed=seed)
    oracle = env.unwrapped.oracle
    p0_intro, p1_intro = intro_frames(oracle, cfg, countdown=countdown)

    done = False
    while not done:
        action = act_p0(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    data = env.episode_data()

    # ramp chute on every real gameplay frame too (matches compose_frame's
    # unconditional per-frame draw_ramp_iso call in the viewer -- it fades
    # away on its own once the puck's deep enough in the field).
    puck_ys = [s["puck"][1] for s in data["states"]]
    data["frames_p0"] = np.stack(
        [_with_ramp_iso(f, cfg, 0, y) for f, y in zip(data["frames_p0"], puck_ys)])
    data["frames_p1"] = np.stack(
        [_with_ramp_iso(f, cfg, 1, y) for f, y in zip(data["frames_p1"], puck_ys)])

    n_intro = len(p0_intro)
    data["frames_p0"] = np.concatenate([np.stack(p0_intro), data["frames_p0"]], axis=0)
    data["frames_p1"] = np.concatenate([np.stack(p1_intro), data["frames_p1"]], axis=0)
    data["actions"] = np.concatenate(
        [np.zeros((n_intro, 2), dtype=np.float32), data["actions"]], axis=0)
    data["actions_p1"] = np.concatenate(
        [np.zeros((n_intro, 2), dtype=np.float32), data["actions_p1"]], axis=0)
    return data


def save_view(frames: np.ndarray, actions: np.ndarray | None, player: str, outcome: str,
              path: Path) -> dict:
    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    payload = {"frames": frames_t, "outcome": outcome, "view": "iso", "player": player}
    if actions is not None:
        payload["actions"] = torch.from_numpy(actions)
        payload["actions_wasd"] = torch.from_numpy(actions_to_wasd(actions))
    torch.save(payload, path)
    return {"file": path.name, "length": frames_t.shape[0] - 1}


def save_game(data: dict, cfg: ArenaConfig, score_before: tuple[int, int],
              score_after: tuple[int, int], match_id: int, game_no: int, matches_dir: Path,
              pad_height: int, hold_frames: int) -> dict:
    """Save one game's two views as separate clips; return its manifest entry.

    `frames` has T+1 entries (the intro frames from intro_frames(), then the
    reset-time frame, then one per real step) plus `hold_frames` repeats of
    the final frame (see stamp_frames), against `actions`'s T padded with a
    no-op per intro frame and per hold frame -- neither is a real env
    transition (the intro's puck position is faked, the hold is a literal
    frame repeat), so frame[t]/action[t] together explaining the transition
    to frame[t+1] holds uniformly across the whole clip, the standard
    video/action alignment.
    """
    outcome = data["outcome"]
    p0_frames = stamp_frames(data["frames_p0"], cfg, score_before, score_after, outcome,
                              pad_height, hold_frames)
    p1_frames = stamp_frames(data["frames_p1"], cfg, score_before[::-1], score_after[::-1],
                              outcome, pad_height, hold_frames)
    actions = np.concatenate(
        [data["actions"], np.zeros((hold_frames, 2), dtype=np.float32)], axis=0)
    actions_p1 = np.concatenate(
        [data["actions_p1"], np.zeros((hold_frames, 2), dtype=np.float32)], axis=0)

    p0_entry = save_view(p0_frames, actions, "p0", outcome,
                          matches_dir / f"match_{match_id:05d}_game{game_no}_p0.pt")
    p1_entry = save_view(p1_frames, actions_p1, "p1", outcome,
                          matches_dir / f"match_{match_id:05d}_game{game_no}_p1.pt")
    return {"game": game_no, "outcome": outcome, "p0": p0_entry, "p1": p1_entry}


def play_match(env: MatchDataRecorder, act_p0, cfg: ArenaConfig, seed: int, match_id: int,
                matches_dir: Path, pad_height: int, hold_frames: int, best_of: int, win_games: int):
    p0_wins = p1_wins = 0
    game_no = 1
    game_entries = []
    while p0_wins < win_games and p1_wins < win_games and game_no <= best_of:
        data = play_game(env, act_p0, seed=seed + game_no, cfg=cfg, countdown=(game_no == 1))
        score_before = (p0_wins, p1_wins)
        if data["outcome"] == "goal":
            p0_wins += 1
        elif data["outcome"] == "conceded":
            p1_wins += 1
        score_after = (p0_wins, p1_wins)
        game_entries.append(
            save_game(data, cfg, score_before, score_after, match_id, game_no, matches_dir,
                      pad_height, hold_frames))
        game_no += 1
    return game_entries, (p0_wins, p1_wins)


def collect(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    matches_dir = out_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)

    cfg = ArenaConfig(n_mallets=2, frame_size=args.frame_size)
    hold_frames = max(1, round(args.hold_secs / cfg.dt))
    checkpoints = discover_checkpoints(args.checkpoints_glob)
    print(f"shard {args.shard}: cycling {len(checkpoints)} checkpoint(s) every "
          f"{args.switch_every} matches: {checkpoints}", flush=True)
    ppo = load_policy(checkpoints[0], cfg)
    env = MatchDataRecorder(AirHockeyMatchEnv(cfg=cfg), view="iso")
    # Separate random streams per player (and per shard), so the two players' random moves are
    # independent of each other and of the env seed.
    shaping = dict(random_frac=args.random_frac, hold_min=args.random_hold_min,
                   hold_max=args.random_hold_max, wasd_only=args.wasd_actions)
    act_p0 = ActionShaper(lambda obs: ppo.act(obs, deterministic=False),
                          np.random.default_rng([args.seed, args.shard, 0]), **shaping)
    env.unwrapped.opponent_policy = ActionShaper(lambda obs: ppo.act(obs, deterministic=False),
                                                 np.random.default_rng([args.seed, args.shard, 1]), **shaping)
    current_checkpoint = checkpoints[0]

    manifest_path = out_dir / f"manifest_{args.shard:04d}.jsonl"
    t0 = time.time()
    with open(manifest_path, "a") as manifest_f:
        for i in range(args.num_matches):
            match_id = args.start_idx + i

            if match_id % args.switch_every == 0:
                # re-glob (not just re-index) so a checkpoint written mid-run
                # by a still-training job is picked up too, and always
                # reload (even if the path is unchanged) since that file's
                # weights may have been overwritten in place since last load.
                checkpoints = discover_checkpoints(args.checkpoints_glob)
                cycle_idx = (match_id // args.switch_every) % len(checkpoints)
                current_checkpoint = checkpoints[cycle_idx]
                ppo.load(current_checkpoint)
                print(f"shard {args.shard}: match {match_id} -> {current_checkpoint}", flush=True)

            seed = args.seed + match_id * (args.best_of + 1)
            game_entries, (p0_wins, p1_wins) = play_match(
                env, act_p0, cfg, seed, match_id, matches_dir, args.pad_height, hold_frames,
                args.best_of, args.win_games)

            manifest_f.write(json.dumps({
                "match_id": match_id, "checkpoint": current_checkpoint, "seed": seed,
                "final_score": [p0_wins, p1_wins], "num_games": len(game_entries),
                "action_shaping": shaping,
                "games": game_entries,
            }) + "\n")
            manifest_f.flush()

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"shard {args.shard}: {i + 1}/{args.num_matches} matches "
                      f"({elapsed / (i + 1):.1f}s/match)", flush=True)


if __name__ == "__main__":
    # Defaults live in configs/data/collect.yml; every flag left out keeps the config's value
    # (argparse defaults are None, which wm.utils.merge_config skips).
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/data/collect.yml")
    ap.add_argument("--out-dir", type=str, required=True,
                     help="collection directory: manifest_<shard>.jsonl files + matches/")
    ap.add_argument("--checkpoints-glob", type=str, nargs="+",
                     help="agent checkpoints to alternate between (glob patterns or paths), re-globbed "
                          "and reloaded every --switch-every matches")
    ap.add_argument("--switch-every", type=int)
    ap.add_argument("--num-matches", type=int)
    ap.add_argument("--start-idx", type=int)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--best-of", type=int, help="at most this many games per match")
    ap.add_argument("--win-games", type=int, help="goals needed to win a match")
    ap.add_argument("--wasd-actions", action=argparse.BooleanOptionalAction,
                     help="execute actions as the W/A/S/D keys they're recorded as (--no-wasd-actions "
                          "records the agents' continuous thrust, rounded to keys only in the labels)")
    ap.add_argument("--random-frac", type=float,
                     help="average fraction of steps each player holds a random key combination")
    ap.add_argument("--random-hold-min", type=int, help="min frames a random move is held")
    ap.add_argument("--random-hold-max", type=int, help="max frames a random move is held")
    ap.add_argument("--frame-size", type=int, help="board render resolution before the header bar")
    ap.add_argument("--pad-height", type=int,
                     help="pad stamped frames to this height, a multiple of the codec's 32x downsampling")
    ap.add_argument("--hold-secs", type=float, help="how long each game's final frame is held")
    raw = ap.parse_args()
    collect(merge_config(raw.config, raw))
