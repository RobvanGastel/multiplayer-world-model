"""
Reusable building blocks for rendering a full striker-vs-defender match as a
GIF with all four views (flat/iso x P0/P1 POV), laid out two columns wide:

    +----------------+----------------+
    |  P0   a - b    |  P1   b - a    |   <- each column's own "you - opp"
    +----------------+----------------+
    |  flat · P0     |  flat · P1     |
    +----------------+----------------+
    |  iso  · P0     |  iso  · P1     |
    +----------------+----------------+
"""
from __future__ import annotations
import argparse
import functools
import math

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wm.env.arena import ArenaConfig, ArenaOracle
from wm.env.policies import policy_strike_to_goal, DEFEND_VARIANTS, make_defend_variant
from wm.env.render_iso import render_iso, _Cam
from wm.env.render import save_gif
from wm.env.match import AirHockeyMatchEnv, agent_observation, DROP_SPEED_MIN_FRAC, DROP_SPEED_MAX_FRAC
from wm.policies.ppo import PPO


TILT_DEG = 35.0
_FONT_FRAC = {"countdown": 13 / 64, "banner": 7 / 64, "ui": 9 / 64}

# orange outline on all on-screen text, for visibility
_OUTLINE_COLOR = (255, 178, 90)  
WIN_GAMES = 3
BEST_OF = 5
ROUND_SECONDS = 10 # shot clock: neither side scores -> the round is a draw

_FLAT_FIELD_COLOR = (26, 30, 42)
_ISO_FLOOR_COLOR = (34, 39, 54)

RAMP_SIZE_FRAC = 3.5 # ramp square side, in puck radii
DROP_START_FRAC = 2.5 # how far outside the field (in puck radii) the puck visually starts
GOAL_POCKET_MARGIN_FRAC = 0.08
GOAL_FOLLOW_THROUGH_FRAMES = 8  # extra visual-only frames after a score


@functools.lru_cache(maxsize=None)
def _fonts_for(frame_size):
    return {name: ImageFont.load_default(size=max(1, round(frame_size * frac)))
            for name, frac in _FONT_FRAC.items()}


def _ramp_polygon_world(cfg, puck_y=None):
    """A square chute straddling the centerline, sitting just outside the
    field above y=0 — same footprint used for its render and for the puck's
    pre-roll path in drop_preroll_frames()."""
    side = cfg.puck_radius * RAMP_SIZE_FRAC
    midx = cfg.width * 0.5
    top = -side
    bottom = 0.0 if puck_y is None else min(0.0, puck_y - cfg.puck_radius * 1.15)
    if bottom <= top:
        return None
    return [(midx - side / 2, top), (midx + side / 2, top),
            (midx + side / 2, bottom), (midx - side / 2, bottom)]


def draw_ramp_flat(draw, cfg, px, puck_y=None):
    poly = _ramp_polygon_world(cfg, puck_y)
    if poly is None:
        return
    sc, ox, oy = ArenaOracle.flat_transform(cfg, px, GOAL_POCKET_MARGIN_FRAC)
    pts = [(ox + x * sc, oy + y * sc) for x, y in poly]
    draw.polygon(pts, fill=_FLAT_FIELD_COLOR, outline=(60, 66, 86), width=2)


def draw_ramp_iso(draw, cfg, agent_idx, px, puck_y=None):
    poly = _ramp_polygon_world(cfg, puck_y)
    if poly is None:
        return
    cam = _Cam(cfg, agent_idx, px, px, tilt_deg=TILT_DEG)
    pts = [cam.project(x, y)[:2] for x, y in poly]
    draw.polygon(pts, fill=_ISO_FLOOR_COLOR, outline=(60, 66, 86), width=2)


def drop_puck(o, rng, speed=None):
    cfg = o.cfg
    if speed is None:
        frac = rng.uniform(DROP_SPEED_MIN_FRAC, DROP_SPEED_MAX_FRAC)
        speed = frac * cfg.puck_max_speed
    o.puck.position = (cfg.width * 0.5, cfg.puck_radius * 1.3)
    o.puck.velocity = (0.0, speed)
    return speed


def drop_preroll_frames(o, snap, speed):
    cfg = o.cfg
    midx = cfg.width * 0.5
    entry_y = cfg.puck_radius * 1.3
    start_y = -cfg.puck_radius * DROP_START_FRAC
    dy = max(1.0, speed * cfg.dt)
    n = max(1, int(np.ceil((entry_y - start_y) / dy)))

    saved_pos, saved_vel = o.puck.position, o.puck.velocity
    frames = []
    for i in range(n):
        y = min(entry_y, start_y + dy * i)
        o.puck.position = (midx, y)
        frames.append(snap())
    o.puck.position, o.puck.velocity = saved_pos, saved_vel
    return frames


def goal_follow_through_frames(o, snap):
    cfg = o.cfg
    if o.outcome not in ("goal", "conceded"):
        return []
    vx, vy = o.puck.velocity.x, o.puck.velocity.y
    speed = math.hypot(vx, vy)
    if speed < 1.0:
        return []

    vx = abs(vx) * (1.0 if o.outcome == "goal" else -1.0)
    pocket_depth = cfg.width * GOAL_POCKET_MARGIN_FRAC * 0.85  # clearance from the very edge
    dt = cfg.dt
    x0, y0 = o.puck.position.x, o.puck.position.y
    frames = []
    for i in range(1, GOAL_FOLLOW_THROUGH_FRAMES + 1):
        t = i * dt
        if speed * t > pocket_depth:
            break
        o.puck.position = (x0 + vx * t, y0 + vy * t)
        frames.append(snap())
    return frames


def load_policy(checkpoint_path: str, cfg: ArenaConfig) -> PPO:
    """Build a PPO whose network shapes match AirHockeyMatchEnv(cfg) and load
    trained weights into it.."""
    dummy = gym.vector.SyncVectorEnv([lambda: AirHockeyMatchEnv(cfg=cfg)])
    ppo_cfg = argparse.Namespace(seed=0, torch_deterministic=True, cuda=False,
                                  hidden_dim=128, learning_rate=2.5e-4,
                                  action_mean_clamp=2.0, action_logstd_range=(-2.0, 0.5))
    ppo = PPO(dummy, cfg=ppo_cfg)
    ppo.load(checkpoint_path)
    ppo.eval()
    dummy.close()
    return ppo


def policy_trained(oracle, idx, ppo: PPO, deterministic: bool = True, mirror: bool = False):
    """Same (oracle, idx) -> act calling convention as the scripted
    policy_* functions, so it can be dropped in as the striker (mirror=False)
    or, for self-play, as the opponent (mirror=True)."""
    def act():
        obs = agent_observation(oracle, mallet_idx=idx, mirror=mirror)
        ax, ay = ppo.act(obs, deterministic=deterministic)
        if mirror:
            ax = -ax
        return (float(ax), float(ay))
    return act


def play_round(cfg, seed, rng, drop_speed=None, striker=None, defender=None):
    o = ArenaOracle(cfg, seed=seed)
    strike = (striker or policy_strike_to_goal)(o, 0)
    if cfg.n_mallets != 2:
        defend = None
    elif defender:
        defend = defender(o, 1)
    else:
        # sample a defensive style from DEFEND_VARIANTS each round, same as
        # AirHockeyMatchEnv.reset()
        variant = o.rng.choice(list(DEFEND_VARIANTS.keys()))
        defend = make_defend_variant(o, 1, side="right", variant=variant)

    def snap():
        return (o.render_flat(margin_frac=GOAL_POCKET_MARGIN_FRAC),
                render_iso(o, agent_idx=0), render_iso(o, agent_idx=1),
                o.puck.position.y)

    speed = drop_puck(o, rng, drop_speed)
    frames = drop_preroll_frames(o, snap, speed)
    frames.append(snap())
    while not o.done:
        acts = [strike()] + ([defend()] if defend else [])
        o.step(acts)
        frames.append(snap())
    frames += goal_follow_through_frames(o, snap)
    # "goal" (striker reaches the far goal) -> P0 scores. "conceded" (own
    # goal) -> P1 scores. "timeout" -> a draw, no points.
    winner = {"goal": 0, "conceded": 1}.get(o.outcome)
    return frames, winner, speed, o.outcome


_TEXT_COLOUR = (255, 213, 140)


def panel_frame(arr: np.ndarray, cfg, header_text: str, header_color=_TEXT_COLOUR,
                 overlay_text: str = "", overlay_font=None) -> Image.Image:
    """One view with its own header bar above it (score, centered), and --
    for the countdown/banner beats -- centered overlay text drawn straight
    on the frame with no background box. The single-view building block
    compose_frame's 2x2 grid assembles four of (via panel() below); a
    standalone recorded clip (not the composed display grid) uses this
    directly so both share the exact same score/banner styling."""
    S = cfg.frame_size
    fonts = _fonts_for(S)
    header_h = max(fonts["ui"].size + 2, round(S * 11 / 64))
    overlay_font = overlay_font or fonts["banner"]

    im = Image.fromarray(arr, "RGB")
    if overlay_text:
        dd = ImageDraw.Draw(im, "RGBA")
        l, t, r, b = dd.textbbox((0, 0), overlay_text, font=overlay_font, stroke_width=1)
        tw, th = r - l, b - t
        cx, cy = S / 2, S / 2
        dd.text((cx - tw / 2 - l, cy - th / 2 - t), overlay_text, fill=(255, 236, 140, 255),
                 font=overlay_font, stroke_width=1, stroke_fill=_OUTLINE_COLOR)

    out = Image.new("RGB", (S, header_h + S), (10, 11, 15))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, S, header_h], fill=(22, 25, 34))
    tw = d.textlength(header_text, font=fonts["ui"])
    d.text((S / 2 - tw / 2, 1), header_text, fill=header_color, font=fonts["ui"],
           stroke_width=1, stroke_fill=_OUTLINE_COLOR)
    out.paste(im, (0, header_h))
    return out


def compose_frame(views, cfg, px, score, countdown="", banner="", speed_label=""):
    """views = (flat0, flat1, iso0, iso1)
    Each of the 4 views (flat/iso x P0/P1) gets its own scoreboard directly
    above it, score centered, and — for the countdown/banner beats — its own
    centered overlay text drawn straight on the frame with no background box."""
    flat0, iso0, iso1, puck_y = views
    flat1 = np.fliplr(flat0).copy()

    S = cfg.frame_size
    fonts = _fonts_for(S)

    # header/speed-bar/gap sizing, tuned by eye.
    gap = max(1, round(S * 1 / 64))
    header_h = max(fonts["ui"].size + 2, round(S * 11 / 64))
    speed_h = max(fonts["ui"].size + 2, round(S * 10 / 64)) if speed_label else 0
    W = S * 2 + gap
    H = speed_h + (header_h + S) * 2 + gap
    lo = Image.new("RGB", (W, H), (10, 11, 15))
    d = ImageDraw.Draw(lo)

    if speed_label:
        tw = d.textlength(speed_label, font=fonts["ui"])
        d.text((W / 2 - tw / 2, 0), speed_label, fill=(170, 178, 200), font=fonts["ui"],
               stroke_width=1, stroke_fill=_OUTLINE_COLOR)

    p0_score, p1_score = score
    p0_text = f"{p0_score}-{p1_score}"
    p1_text = f"{p1_score}-{p0_score}"
    overlay_text = countdown or banner
    overlay_font = fonts["countdown"] if countdown else fonts["banner"]

    def panel(arr, col, row, is_flat, agent_idx, header_text, header_color):
        x = col * (S + gap)
        y = speed_h + row * (S + header_h + gap)

        im = Image.fromarray(arr, "RGB")
        dd = ImageDraw.Draw(im, "RGBA")
        if is_flat:
            draw_ramp_flat(dd, cfg, S, puck_y)
        else:
            draw_ramp_iso(dd, cfg, agent_idx, S, puck_y)
        arr_with_ramp = np.array(im, dtype=np.uint8)

        panel_im = panel_frame(arr_with_ramp, cfg, header_text, header_color,
                                overlay_text, overlay_font)
        lo.paste(panel_im, (x, y))

    panel(flat0, 0, 0, True, 0, p0_text, _TEXT_COLOUR)
    panel(flat1, 1, 0, True, 1, p1_text, _TEXT_COLOUR)
    panel(iso0, 0, 1, False, 0, p0_text, _TEXT_COLOUR)
    panel(iso1, 1, 1, False, 1, p1_text, _TEXT_COLOUR)

    out_w = round(W * px / S)
    out_h = round(H * px / S)
    return lo.resize((out_w, out_h), Image.NEAREST)


def countdown_frames(views, cfg, px, score, fps):
    """3, 2, 1, GO! held over the static pre-drop scene."""
    seq = [("3", 0.6), ("2", 0.6), ("1", 0.6), ("GO!", 0.5)]
    out = []
    for text, secs in seq:
        im = compose_frame(views, cfg, px, score, countdown=text)
        out += [im] * max(1, int(fps * secs))
    return out


def play_match(cfg, seed, px, fps, hold_secs, striker=None, defender=None,
                win_games=WIN_GAMES, best_of=BEST_OF):
    rng = np.random.default_rng(seed)
    p0_score = p1_score = 0
    game_no = 1
    all_frames = []

    while p0_score < win_games and p1_score < win_games and game_no <= best_of:
        frames, winner, speed, outcome = play_round(
            cfg, seed=seed + game_no, rng=rng, striker=striker, defender=defender)

        pre_score = (p0_score, p1_score)
        all_frames += countdown_frames(frames[0], cfg, px, pre_score, fps)
        for v in frames:
            all_frames.append(compose_frame(v, cfg, px, pre_score))

        if winner == 0:
            p0_score += 1
        elif winner == 1:
            p1_score += 1
        # else: draw (shot clock ran out) -- neither score changes
        banner = {"goal": "P0 SCORES", "conceded": "P1 SCORES"}.get(
            outcome, "TIME'S UP - DRAW")
        hold = compose_frame(frames[-1], cfg, px, (p0_score, p1_score), banner=banner)
        all_frames += [hold] * int(fps * hold_secs)
        game_no += 1

    if p0_score > p1_score:
        final_banner = f"P0 WINS {p0_score}-{p1_score}"
    elif p1_score > p0_score:
        final_banner = f"P1 WINS {p1_score}-{p0_score}"
    else:
        final_banner = f"MATCH DRAWN {p0_score}-{p1_score}"
    match_over = compose_frame(frames[-1], cfg, px, (p0_score, p1_score), banner=final_banner)
    all_frames += [match_over] * int(fps * (hold_secs * 1.6))
    return all_frames, (p0_score, p1_score)
