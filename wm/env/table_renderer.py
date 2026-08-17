"""
'table' renderer — render-only 2.5D effects on top of the UNTOUCHED oracle.

This does not change any physics. It re-renders the exact same simulation with
depth cues that make the flat 2D game read as a physical air-hockey table:

  * HOP + SEPARATING SHADOW. The puck gets a fake vertical offset (z) as a
    function of its speed, and its floor shadow stays on the plane — so on a
    hard shot the puck lifts off its shadow. This is a deterministic function of
    puck speed, which makes it a nice world-model test later: a good model
    reproduces the coupling, a bad one leaves the shadow stuck to the puck.
  * BEVELED RAILS. Walls drawn as short 3D rails (lit top + dark face) so the
    playfield looks like a table, not a flat rectangle.
  * SPECULAR HIGHLIGHT. A bright spot on each disc so it reads as rounded.
  * CONTACT SHADOWS. Soft grounding under each body.

Everything here is a function of the oracle's state, added at draw time. Train
the first world model on the baseline flat frames; then, as a deliberate
ablation, train on these frames and check whether the model recovers the
depth cues (the hop-shadow coupling especially) or only the positions.

Usage:
    python table_renderer.py --preview-gif table.gif           # table look
    python table_renderer.py --compare-gif compare.gif         # baseline | table
    python table_renderer.py --hidden-field --preview-gif t.gif
"""

from __future__ import annotations
import math
import argparse
import numpy as np
from PIL import Image

from arena_oracle import (
    ArenaConfig, ArenaOracle,
    policy_strike_to_goal, policy_chase_puck, policy_evade, policy_random,
)


# --------------------------------------------------------------------------
# drawing helpers (float-center discs, soft edges)
# --------------------------------------------------------------------------

def _disc(img, cx, cy, r, color, alpha=1.0):
    S = img.shape[0]
    r = max(1.0, r)
    x0, x1 = max(0, int(cx - r - 1)), min(S, int(cx + r + 2))
    y0, y1 = max(0, int(cy - r - 1)), min(S, int(cy + r + 2))
    col = np.array(color, dtype=np.float32)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            d = math.hypot(xx - cx, yy - cy)
            if d <= r:
                # soft 1px edge
                a = alpha * min(1.0, (r - d))
                a = max(0.0, min(1.0, a if (r - d) < 1 else alpha))
                img[yy, xx] = (1 - a) * img[yy, xx] + a * col


def _soft_shadow(img, cx, cy, rx, ry, strength=0.5):
    """Flattened ellipse shadow on the floor."""
    S = img.shape[0]
    x0, x1 = max(0, int(cx - rx - 1)), min(S, int(cx + rx + 2))
    y0, y1 = max(0, int(cy - ry - 1)), min(S, int(cy + ry + 2))
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            dx = (xx - cx) / (rx + 1e-6)
            dy = (yy - cy) / (ry + 1e-6)
            d = dx * dx + dy * dy
            if d <= 1.0:
                a = strength * (1.0 - d) ** 1.5
                img[yy, xx] = (1 - a) * img[yy, xx]


# --------------------------------------------------------------------------
# table renderer
# --------------------------------------------------------------------------

MALLET_COLORS = [(90, 160, 240), (240, 110, 110)]
MALLET_HI = [(200, 225, 255), (255, 200, 200)]


def render_table(oracle: ArenaOracle) -> np.ndarray:
    cfg = oracle.cfg
    S = cfg.frame_size
    sx = S / cfg.width
    sy = S / cfg.height
    img = np.zeros((S, S, 3), dtype=np.float32)

    # --- table surface with a subtle vignette ---
    img[:] = (26, 30, 42)
    # playfield inset panel
    pad = int(3)
    img[pad:S - pad, pad:S - pad] = (34, 39, 54)

    # --- beveled rails (lit top edge + dark inner face) ---
    rail = int(max(2, S * 0.05))
    lit = np.array((70, 80, 104), np.float32)
    dark = np.array((16, 18, 26), np.float32)
    # top rail
    img[0:rail, :] = lit
    img[rail:rail + 2, :] = dark
    # bottom rail
    img[S - rail:S, :] = lit * 0.7
    img[S - rail - 2:S - rail, :] = dark
    # left rail
    img[:, 0:rail] = lit * 0.85
    img[:, rail:rail + 2] = dark
    # right rail (with goal gap)
    g = oracle.goal
    gy0, gy1 = int(g["y0"] * sy), int(g["y1"] * sy)
    img[:, S - rail:S] = lit * 0.85
    img[:, S - rail - 2:S - rail] = dark
    # carve the goal mouth into the right rail
    img[gy0:gy1, S - rail:S] = (46, 96, 66)
    img[gy0:gy1, S - rail - 2:S - rail] = (90, 170, 120)

    # --- center line + face-off circle, faint ---
    cxp = int(S * 0.5)
    img[rail:S - rail, cxp:cxp + 1] = img[rail:S - rail, cxp:cxp + 1] * 0.7 + np.array((60, 68, 88)) * 0.3

    # --- bodies: shadow on floor, then hopped body, then highlight ---
    # puck hop as a function of speed
    pv = oracle.puck.velocity
    pspeed = math.hypot(pv.x, pv.y)
    hop = min(1.0, pspeed / cfg.puck_max_speed)          # 0..1
    z = hop * (S * 0.06)                                   # vertical lift in px
    p = oracle.puck.position
    pxp, pyp = p.x * sx, p.y * sy
    pr = cfg.puck_radius * sx

    # shadow stays on the floor; shrinks + softens as puck lifts
    _soft_shadow(img, pxp, pyp + pr * 0.3, pr * (1.2 - 0.4 * hop),
                 pr * (0.6 - 0.2 * hop), strength=0.55 - 0.25 * hop)
    # puck drawn lifted by z
    _disc(img, pxp, pyp - z, pr, (240, 220, 90))
    # specular highlight (upper-left)
    _disc(img, pxp - pr * 0.35, pyp - z - pr * 0.35, pr * 0.35, (255, 248, 200))

    # mallets
    for i, b in enumerate(oracle.mallets):
        bxp, byp = b.position.x * sx, b.position.y * sy
        br = cfg.mallet_radius * sx
        _soft_shadow(img, bxp, byp + br * 0.3, br * 1.15, br * 0.55, strength=0.5)
        _disc(img, bxp, byp, br, MALLET_COLORS[i % len(MALLET_COLORS)])
        _disc(img, bxp - br * 0.3, byp - br * 0.3, br * 0.35, MALLET_HI[i % len(MALLET_HI)])

    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# rollout + preview
# --------------------------------------------------------------------------

def rollout_frames(cfg, seed, renderer_fn, policies="strike"):
    o = ArenaOracle(cfg, seed=seed)
    if policies == "random":
        pols = [policy_random(o, i) for i in range(cfg.n_mallets)]
    elif cfg.mode == "pursuit" and cfg.n_mallets >= 2:
        pols = [policy_chase_puck(o, 0), policy_evade(o, 1)]
    else:
        pols = [policy_strike_to_goal(o, i) for i in range(cfg.n_mallets)]

    frames = [renderer_fn(o)]
    while not o.done:
        acts = [pl() for pl in pols]
        o.step(acts)
        frames.append(renderer_fn(o))
    return frames, o.outcome


def render_baseline(oracle):
    # baseline flat renderer, straight from the oracle (untouched)
    return oracle._render_flat()


def save_gif(frames, path, scale=4, fps=30):
    imgs = [Image.fromarray(f, "RGB").resize(
        (f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)


def save_compare_gif(cfg, seed, path, scale=4, fps=30):
    """Baseline (left) and table (right) on the SAME seed, side by side."""
    base, _ = rollout_frames(cfg, seed, render_baseline)
    tab, _ = rollout_frames(cfg, seed, render_table)
    n = min(len(base), len(tab))
    gap = 6
    imgs = []
    for i in range(n):
        S = base[i].shape[0]
        canvas = np.full((S, S * 2 + gap, 3), 12, np.uint8)
        canvas[:, :S] = base[i]
        canvas[:, S + gap:] = tab[i]
        im = Image.fromarray(canvas, "RGB").resize(
            ((S * 2 + gap) * scale, S * scale), Image.NEAREST)
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview-gif", type=str, default="")
    ap.add_argument("--compare-gif", type=str, default="")
    ap.add_argument("--mode", type=str, default="airhockey", choices=["airhockey", "pursuit"])
    ap.add_argument("--mallets", type=int, default=1)
    ap.add_argument("--hidden-field", action="store_true")
    ap.add_argument("--frame-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    if args.mode == "pursuit" and args.mallets < 2:
        args.mallets = 2

    cfg = ArenaConfig(mode=args.mode, n_mallets=args.mallets,
                      hidden_field=args.hidden_field, frame_size=args.frame_size)

    if args.compare_gif:
        save_compare_gif(cfg, args.seed, args.compare_gif)
        print(f"wrote {args.compare_gif} (left=baseline flat, right=table)")
    if args.preview_gif:
        frames, outcome = rollout_frames(cfg, args.seed, render_table)
        save_gif(frames, args.preview_gif)
        print(f"wrote {args.preview_gif} ({len(frames)} frames, outcome={outcome})")
    if not args.compare_gif and not args.preview_gif:
        print("nothing to do — pass --preview-gif or --compare-gif")


if __name__ == "__main__":
    main()
