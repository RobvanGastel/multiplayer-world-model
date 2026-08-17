"""
Best-of-5 match GIF with BOTH views (2D flat + smooth angled), a scoreboard
(first to 3 wins), and a WASD key HUD showing the striker's pressed keys.

Composes per frame:  [ scoreboard bar ]
                     [ 2D flat view | angled view ]
                     [ WASD key pad ]

Physics from the untouched oracle; angled panel via smooth_angled.render_smooth,
2D panel via a compact flat draw. Action->keys: ax>0 D, ax<0 A, ay>0 S, ay<0 W.

Usage:
    python match_both_views.py --seed 1 --out match_both.gif
"""
from __future__ import annotations
import argparse
import numpy as np
from PIL import Image, ImageDraw

from arena_oracle import (
    ArenaConfig, ArenaOracle,
    policy_strike_to_goal, policy_defend_goal,
)
from smooth_angled import render_smooth, _snapshot

KEY_CELLS = {"w": (1, 0), "a": (0, 1), "s": (1, 1), "d": (2, 1)}


def action_to_keys(ax, ay, thresh=0.15):
    ks = set()
    if ax > thresh: ks.add("d")
    if ax < -thresh: ks.add("a")
    if ay > thresh: ks.add("s")
    if ay < -thresh: ks.add("w")
    return ks


def draw_flat_panel(snap, px):
    """Compact top-down 2D panel at px x px."""
    cfg = snap.cfg
    im = Image.new("RGB", (px, px), (12, 14, 20))
    d = ImageDraw.Draw(im, "RGBA")
    sc = min(px / cfg.width, px / cfg.height)
    ox = (px - cfg.width * sc) / 2
    oy = (px - cfg.height * sc) / 2
    def X(x): return ox + x * sc
    def Y(y): return oy + y * sc
    # playfield
    d.rectangle([X(0), Y(0), X(cfg.width), Y(cfg.height)], fill=(26, 30, 42))
    gy0 = (cfg.height - cfg.height * 0.5) / 2
    gy1 = (cfg.height + cfg.height * 0.5) / 2
    # center line
    d.line([(X(cfg.width / 2), Y(0)), (X(cfg.width / 2), Y(cfg.height))], fill=(90, 98, 124), width=2)
    # goals (both)
    d.rectangle([X(cfg.width) - 4, Y(gy0), X(cfg.width), Y(gy1)], fill=(90, 200, 130))
    d.rectangle([X(0), Y(gy0), X(0) + 4, Y(gy1)], fill=(210, 170, 90))
    # bodies (small, matching the open-field look)
    pr = cfg.puck_radius * 0.5 * sc
    d.ellipse([X(snap.puck.position.x) - pr, Y(snap.puck.position.y) - pr,
               X(snap.puck.position.x) + pr, Y(snap.puck.position.y) + pr], fill=(240, 220, 90))
    mcol = [(90, 160, 240), (240, 110, 110)]
    for i, m in enumerate(snap.mallets):
        mr = cfg.mallet_radius * 0.5 * sc
        d.ellipse([X(m.position.x) - mr, Y(m.position.y) - mr,
                   X(m.position.x) + mr, Y(m.position.y) + mr], fill=mcol[i % 2])
    return im


def compose_frame(snap, px, you, opp, game_no, keys, action, banner=""):
    gap = 8
    bar_h = 46
    hud_h = 84
    flat = draw_flat_panel(snap, px)
    ang = render_smooth(snap, 0, out_px=px, tilt_deg=35.0)
    W = px * 2 + gap
    H = bar_h + px + hud_h
    out = Image.new("RGB", (W, H), (14, 16, 22))
    d = ImageDraw.Draw(out)

    # scoreboard bar
    d.rectangle([0, 0, W, bar_h], fill=(22, 25, 34))
    d.text((W * 0.12, 8), "YOU", fill=(150, 200, 255))
    d.text((W * 0.14, 22), str(you), fill=(90, 160, 240))
    mid = banner if banner else f"game {game_no} of 5   first to 3"
    tw = d.textlength(mid)
    d.text((W / 2 - tw / 2, 16), mid, fill=(200, 206, 224))
    d.text((W * 0.84, 8), "DEF", fill=(255, 170, 170))
    d.text((W * 0.86, 22), str(opp), fill=(240, 110, 110))

    # the two views
    out.paste(flat, (0, bar_h))
    out.paste(ang, (px + gap, bar_h))
    # view labels
    d.text((8, bar_h + 6), "2D", fill=(150, 158, 180))
    d.text((px + gap + 8, bar_h + 6), "angled", fill=(150, 158, 180))

    # WASD HUD centered under the views
    cell = 26
    cgap = 4
    pad_w = cell * 3 + cgap * 2
    x0 = W / 2 - pad_w / 2
    y0 = bar_h + px + 14
    for k, (cx, cy) in KEY_CELLS.items():
        kx = x0 + cx * (cell + cgap)
        ky = y0 + cy * (cell + cgap)
        lit = k in keys
        d.rectangle([kx, ky, kx + cell, ky + cell],
                    fill=(90, 160, 240) if lit else (30, 34, 46),
                    outline=(150, 200, 255) if lit else (70, 76, 96), width=2)
        d.text((kx + cell / 2 - 4, ky + cell / 2 - 6), k.upper(),
               fill=(12, 20, 34) if lit else (120, 128, 150))
    d.text((x0 + pad_w + 14, y0 + 18),
           f"action: {action[0]:+.1f}, {action[1]:+.1f}", fill=(170, 178, 200))
    return out


def play_round(cfg, seed, weak_defender=False):
    o = ArenaOracle(cfg, seed=seed)
    strike = policy_strike_to_goal(o, 0)
    defend = policy_defend_goal(o, 1, side="right",
                                noise=0.6 if weak_defender else 0.15)
    frames = []
    while not o.done:
        a0 = strike()
        o.step([a0, defend()])
        frames.append((_snapshot(o), action_to_keys(a0[0], a0[1]), a0))
    winner = "you" if o.outcome == "goal" else "opp"
    return frames, winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="match_both.gif")
    ap.add_argument("--px", type=int, default=200)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--win", type=int, default=3)
    ap.add_argument("--games", type=int, default=5)
    args = ap.parse_args()

    cfg = ArenaConfig(mode="airhockey", n_mallets=2)
    you, opp, game_no = 0, 0, 1
    weak_sched = [False, True, False, True, False, True, False]
    all_frames = []

    while you < args.win and opp < args.win and game_no <= args.games:
        weak = weak_sched[(game_no - 1) % len(weak_sched)]
        frames, winner = play_round(cfg, seed=args.seed + game_no, weak_defender=weak)
        if winner == "you": you += 1
        else: opp += 1
        # render round frames with the running score BEFORE this goal
        sy = you - (1 if winner == "you" else 0)
        so = opp - (1 if winner == "opp" else 0)
        for (snap, keys, action) in frames:
            all_frames.append(compose_frame(snap, args.px, sy, so, game_no, keys, action))
        # hold the post-goal moment
        last = frames[-1][0]
        hold = compose_frame(last, args.px, you, opp, game_no, set(), (0.0, 0.0),
                             banner=f"{'YOU' if winner=='you' else 'DEFENDER'} scores!")
        for _ in range(int(args.fps * 0.7)):
            all_frames.append(hold)
        game_no += 1

    # final card
    wtxt = f"YOU WIN {you}-{opp}" if you > opp else f"DEFENDER WINS {opp}-{you}"
    last = frames[-1][0]
    final = compose_frame(last, args.px, you, opp, game_no - 1, set(), (0.0, 0.0), banner=wtxt)
    for _ in range(int(args.fps * 1.6)):
        all_frames.append(final)

    all_frames[0].save(args.out, save_all=True, append_images=all_frames[1:],
                       duration=int(1000 / args.fps), loop=0)
    print(f"wrote {args.out} ({len(all_frames)} frames, final {you}-{opp})")


if __name__ == "__main__":
    main()
