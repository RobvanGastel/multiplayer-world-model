"""
Render the scripted policy playing air-hockey with a WASD key overlay that
flashes which keys the policy is 'pressing' each frame. The player's action
vector (ax, ay) maps to keys: ax>0->D, ax<0->A, ay>0->S, ay<0->W.

Produces a GIF at a comfortable display size. Physics/board come from the
untouched oracle; this only adds the key HUD on top of the flat render.

Usage:
    python play_with_keys.py --seed 3 --out play_keys.gif
    python play_with_keys.py --seed 3 --mallets 2 --out play_keys.gif
"""
from __future__ import annotations
import argparse
import numpy as np
from PIL import Image, ImageDraw

from arena_oracle import (
    ArenaConfig, ArenaOracle,
    policy_strike_to_goal, policy_defend_goal,
)

# key layout (grid cells), each is (col, row) in a 3x2 grid
KEY_CELLS = {"w": (1, 0), "a": (0, 1), "s": (1, 1), "d": (2, 1)}


def action_to_keys(ax, ay, thresh=0.15):
    keys = set()
    if ax > thresh: keys.add("d")
    if ax < -thresh: keys.add("a")
    if ay > thresh: keys.add("s")
    if ay < -thresh: keys.add("w")
    return keys


def render_flat_big(oracle, S_out=300):
    """Render the flat board upscaled to S_out px (nearest), return a PIL image."""
    frame = oracle._render_flat()  # small square frame
    im = Image.fromarray(frame, "RGB").resize((S_out, S_out), Image.NEAREST)
    return im


def add_scoreboard(board_im, you, opp, game_no, banner=""):
    """Return a new image with a scoreboard header bar above the board image."""
    bar_h = 46
    W = board_im.width
    out = Image.new("RGB", (W, board_im.height + bar_h), (14, 16, 22))
    d = ImageDraw.Draw(out)
    # header background
    d.rectangle([0, 0, W, bar_h], fill=(22, 25, 34))
    # You (left, blue)
    d.text((W * 0.14, 8), "YOU", fill=(150, 200, 255))
    d.text((W * 0.16, 22), str(you), fill=(90, 160, 240))
    # center: match state / banner
    mid_txt = banner if banner else f"game {game_no} of 5  ·  first to 3"
    tw = d.textlength(mid_txt)
    d.text((W / 2 - tw / 2, 16), mid_txt, fill=(180, 186, 210))
    # Defender (right, red)
    d.text((W * 0.80, 8), "DEF", fill=(255, 170, 170))
    d.text((W * 0.82, 22), str(opp), fill=(240, 110, 110))
    # paste the board below the bar
    out.paste(board_im, (0, bar_h))
    return out


def draw_key_hud(im, keys, action):
    """Draw a WASD key pad at bottom-left; lit keys are highlighted."""
    d = ImageDraw.Draw(im)
    cell = 26
    gap = 4
    pad_w = cell * 3 + gap * 2
    x0 = 12
    y0 = im.height - (cell * 2 + gap) - 12
    for k, (cx, cy) in KEY_CELLS.items():
        kx = x0 + cx * (cell + gap)
        ky = y0 + cy * (cell + gap)
        lit = k in keys
        fill = (90, 160, 240) if lit else (30, 34, 46)
        outline = (150, 200, 255) if lit else (70, 76, 96)
        d.rectangle([kx, ky, kx + cell, ky + cell], fill=fill, outline=outline, width=2)
        tcol = (12, 20, 34) if lit else (120, 128, 150)
        d.text((kx + cell // 2 - 4, ky + cell // 2 - 6), k.upper(), fill=tcol)
    # action vector readout
    d.text((x0, y0 - 16), f"action: {action[0]:+.1f}, {action[1]:+.1f}",
           fill=(180, 186, 210))
    return im


def play_round(cfg, seed, weak_defender=False):
    """Play one round; return (frames_data, winner). frames_data is a list of
    (oracle_snapshot_frame, keys, action)."""
    o = ArenaOracle(cfg, seed=seed)
    strike = policy_strike_to_goal(o, 0)
    if cfg.n_mallets == 2:
        # a 'weak' defender reacts with more noise so some rounds go to the striker
        # cleanly and others let the puck get past — creates a real 5-game spread
        defend = policy_defend_goal(o, 1, side="right",
                                    noise=0.6 if weak_defender else 0.15)
    else:
        defend = None

    frames = []
    while not o.done:
        a0 = strike()
        acts = [a0] + ([defend()] if defend else [])
        o.step(acts)
        keys = action_to_keys(a0[0], a0[1])
        frames.append((o._render_flat(), keys, a0))
    # winner: striker scores in far goal -> 'you'; own goal or timeout -> 'opp'
    winner = "you" if o.outcome == "goal" else "opp"
    return frames, winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mallets", type=int, default=2)
    ap.add_argument("--out", type=str, default="match_keys.gif")
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    cfg = ArenaConfig(mode="airhockey", n_mallets=args.mallets)

    you, opp, game_no = 0, 0, 1
    all_frames = []
    rng_seed = args.seed
    # pre-decide a mix so the match is a real contest, not 3-0
    # (alternate weak/strong defender to spread wins across ~5 games)
    weak_schedule = [False, True, False, True, False, True, False]

    while you < 3 and opp < 3 and game_no <= 5:
        weak = weak_schedule[(game_no - 1) % len(weak_schedule)]
        frames, winner = play_round(cfg, seed=rng_seed + game_no, weak_defender=weak)
        if winner == "you": you += 1
        else: opp += 1

        # render this round's frames with the CURRENT running score
        for (board_frame, keys, action) in frames:
            im = Image.fromarray(board_frame, "RGB").resize((args.size, args.size), Image.NEAREST)
            im = draw_key_hud(im, keys, action)
            im = add_scoreboard(im, you if winner != "you" else you - 1,
                                opp if winner != "opp" else opp - 1, game_no)
            all_frames.append(im)

        # hold on the post-goal score for ~0.7s
        last_board = Image.fromarray(frames[-1][0], "RGB").resize((args.size, args.size), Image.NEAREST)
        last_board = draw_key_hud(last_board, set(), (0.0, 0.0))
        hold = add_scoreboard(last_board, you, opp, game_no,
                              banner=f"{'YOU' if winner=='you' else 'DEFENDER'} scores!")
        for _ in range(int(args.fps * 0.7)):
            all_frames.append(hold)
        game_no += 1

    # final banner
    winner_txt = f"YOU WIN {you}-{opp}" if you > opp else f"DEFENDER WINS {opp}-{you}"
    final_board = Image.new("RGB", (args.size, args.size), (18, 20, 28))
    final = add_scoreboard(final_board, you, opp, game_no - 1, banner=winner_txt)
    for _ in range(int(args.fps * 1.5)):
        all_frames.append(final)

    all_frames[0].save(args.out, save_all=True, append_images=all_frames[1:],
                       duration=int(1000 / args.fps), loop=0)
    print(f"wrote {args.out} ({len(all_frames)} frames, final {you}-{opp})")


if __name__ == "__main__":
    main()
