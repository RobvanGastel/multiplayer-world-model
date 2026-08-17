"""
Smooth angled renderer (PIL, anti-aliased, supersampled).

Replaces the 64px pixel-loop rasterizer with PIL ImageDraw at high resolution,
which is why this looks like the browser version instead of the jagged GIF.
Adds net/posts goals, an even floor with a subtle grid, and smaller bodies on a
bigger field. Physics comes from the untouched oracle; this is render-only and
reads the same absolute board state, so both agent sides stay consistent.

Usage:
    python smooth_angled.py --seed 0 --both --out both_sides_smooth.gif
    python smooth_angled.py --seed 0 --agent 0 --out one_side.gif
"""
from __future__ import annotations
import argparse, math
import numpy as np
from PIL import Image, ImageDraw

from arena_oracle import (
    ArenaConfig, ArenaOracle,
    policy_strike_to_goal, policy_defend_goal,
)

# body-to-field ratio: smaller bodies on a bigger field. These override the
# oracle's render sizes at DRAW time only (physics radii are unchanged, but we
# draw them smaller to get the "more open space" look you asked for).
PUCK_DRAW_FRAC = 0.147    # 2/3 of previous (0.22)
MALLET_DRAW_FRAC = 0.16   # 2/3 of previous (0.24)


class Cam:
    def __init__(self, cfg, agent_idx, W, Hpx, tilt_deg=35.0):
        self.cfg = cfg; self.W = W; self.H = Hpx
        self.flip = (agent_idx == 1)
        import math as _m
        tilt = _m.radians(tilt_deg)
        comp = _m.cos(tilt)                    # vertical compression of depth
        span = 0.78 * comp
        # pull the near edge up from the very bottom so there's foreground floor:
        # the own goal (near) sits clearly IN FRONT of the player, not at the lip.
        top = 0.5 - span / 2 + 0.02
        self.horizon_y = Hpx * top
        self.near_y = Hpx * (top + span)
        self.near_hw = W * 0.42
        self.far_hw = W * (0.42 * (0.55 + 0.45 * comp))

    def depth(self, x):
        d = x / self.cfg.width
        return (1.0 - d) if self.flip else d

    def project(self, x, y):
        t = self.depth(x)
        pt = t / (1.0 + 0.15 * (1.0 - t))
        py = self.near_y + (self.horizon_y - self.near_y) * pt
        hw = self.near_hw + (self.far_hw - self.near_hw) * pt
        v = (y / self.cfg.height) - 0.5
        px = self.W * 0.5 + v * (2 * hw)
        return px, py, (2 * hw) / self.cfg.height


def _lerp(a, b, t): return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def render_smooth(oracle, agent_idx, out_px=256, ss=3, tilt_deg=40.0):
    """Render one agent's angled view at out_px, supersampled by ss for AA."""
    cfg = oracle.cfg
    W = H = out_px * ss
    cam = Cam(cfg, agent_idx, W, H, tilt_deg=tilt_deg)
    im = Image.new("RGB", (W, H), (16, 18, 26))
    d = ImageDraw.Draw(im, "RGBA")

    gy0 = (cfg.height - cfg.height * 0.5) / 2
    gy1 = (cfg.height + cfg.height * 0.5) / 2

    # ---- floor trapezoid ----
    c = [cam.project(0, 0), cam.project(0, cfg.height),
         cam.project(cfg.width, cfg.height), cam.project(cfg.width, 0)]
    d.polygon([(p[0], p[1]) for p in c], fill=(34, 39, 54))

    # ---- even grid on the floor (depth lines + lateral lines) ----
    grid = (48, 54, 72)
    for gx in range(0, int(cfg.width) + 1, int(cfg.width // 6)):
        a = cam.project(gx, 0); b = cam.project(gx, cfg.height)
        d.line([(a[0], a[1]), (b[0], b[1])], fill=grid, width=max(1, ss))
    for gy in range(0, int(cfg.height) + 1, int(cfg.height // 4)):
        a = cam.project(0, gy); b = cam.project(cfg.width, gy)
        d.line([(a[0], a[1]), (b[0], b[1])], fill=grid, width=max(1, ss))

    # ---- center line (brighter) ----
    m0 = cam.project(cfg.width * 0.5, 0); m1 = cam.project(cfg.width * 0.5, cfg.height)
    d.line([(m0[0], m0[1]), (m1[0], m1[1])], fill=(90, 98, 124), width=max(2, ss * 2))

    # ---- goals as shallow 3D boxes: mouth at the goal line, net set back
    #      ~half a tile beyond the line so the goal reads with depth ----
    tile = cfg.width / 6.0
    setback = tile * 0.5                 # half a tile of goal depth

    def draw_net_goal(board_x, post_col, net_col, outward):
        # Real goal: upright rectangular frame on the goal line (two posts + a
        # flat crossbar), with the net sloping BACK and DOWN from the crossbar to
        # a back bar resting on the ground behind the line.
        back_x = board_x + outward * setback
        # front frame corners (on the goal line) — take (x,y) from projection
        pf0 = cam.project(board_x, gy0); f0 = (pf0[0], pf0[1])
        pf1 = cam.project(board_x, gy1); f1 = (pf1[0], pf1[1])
        pb0 = cam.project(back_x, gy0);  b0 = (pb0[0], pb0[1])
        pb1 = cam.project(back_x, gy1);  b1 = (pb1[0], pb1[1])
        post_h = 9 * ss * pf0[2]          # crossbar height above the ground

        # crossbar top corners (above the front posts)
        t0 = (f0[0], f0[1] - post_h)
        t1 = (f1[0], f1[1] - post_h)

        # floor of the goal (front line -> back bar), faint
        d.polygon([f0, f1, b1, b0], fill=(28, 32, 44))

        # net roof: sloping panel from the crossbar down to the back bar
        # (draw as a filled quad, faint, then mesh lines over it)
        d.polygon([t0, t1, b1, b0], fill=(net_col[0], net_col[1], net_col[2], 70))
        # net mesh — lines along the slope (crossbar -> back bar) at intervals
        nn = 6
        for kk in range(0, nn + 1):
            fx = t0[0] + (t1[0] - t0[0]) * kk / nn
            fy = t0[1] + (t1[1] - t0[1]) * kk / nn
            bx = b0[0] + (b1[0] - b0[0]) * kk / nn
            by = b0[1] + (b1[1] - b0[1]) * kk / nn
            d.line([(fx, fy), (bx, by)], fill=net_col, width=max(1, ss))
        # net mesh — lines across the slope
        for r in (0.33, 0.66):
            ax = t0[0] + (b0[0] - t0[0]) * r; ay = t0[1] + (b0[1] - t0[1]) * r
            cx = t1[0] + (b1[0] - t1[0]) * r; cy = t1[1] + (b1[1] - t1[1]) * r
            d.line([(ax, ay), (cx, cy)], fill=net_col, width=max(1, ss))

        # side net panels (triangle: front post top -> front bottom -> back bottom)
        d.polygon([t0, f0, b0], fill=(net_col[0], net_col[1], net_col[2], 45))
        d.polygon([t1, f1, b1], fill=(net_col[0], net_col[1], net_col[2], 45))

        # FRONT FRAME on top: two posts + flat crossbar (the solid goal frame)
        d.line([f0, t0], fill=post_col, width=max(2, int(ss * 2.6)))   # post 0
        d.line([f1, t1], fill=post_col, width=max(2, int(ss * 2.6)))   # post 1
        d.line([t0, t1], fill=post_col, width=max(2, int(ss * 2.6)))   # crossbar (flat)
        # back bar on the ground
        d.line([b0, b1], fill=post_col, width=max(2, int(ss * 1.8)))
        # back stays (crossbar corners down to back bar) — the frame that holds the net
        d.line([t0, b0], fill=post_col, width=max(2, int(ss * 1.5)))
        d.line([t1, b1], fill=post_col, width=max(2, int(ss * 1.5)))

    far_x = cfg.width if not cam.flip else 0.0
    near_x = 0.0 if not cam.flip else cfg.width
    far_out = +1 if not cam.flip else -1    # far goal opens away from field
    near_out = -1 if not cam.flip else +1   # near goal opens toward the camera
    # far (opponent) goal is deepest -> draw it BEFORE the bodies
    draw_net_goal(far_x, (90, 200, 130), (60, 120, 85, 180), far_out)   # opponent (green)

    # ---- bodies far -> near with soft shadows and highlight ----
    bodies = [("puck", oracle.puck.position, (240, 220, 90),
               cfg.puck_radius * PUCK_DRAW_FRAC)]
    mcol = [(90, 160, 240), (240, 110, 110)]
    for i, b in enumerate(oracle.mallets):
        bodies.append(("mallet", b.position, mcol[i % 2],
                       cfg.mallet_radius * MALLET_DRAW_FRAC))
    bodies.sort(key=lambda t: cam.depth(t[1].x), reverse=True)

    for kind, pos, col, br in bodies:
        px, py, sc = cam.project(pos.x, pos.y)
        rpx = br * (2 * cam.near_hw / cfg.height) * sc
        # shadow
        d.ellipse([px - rpx * 1.15, py + rpx * 0.15, px + rpx * 1.15, py + rpx * 0.75],
                  fill=(0, 0, 0, 110))
        # body
        d.ellipse([px - rpx, py - rpx, px + rpx, py + rpx], fill=col)
        # rim + highlight for a rounded look
        d.ellipse([px - rpx, py - rpx, px + rpx, py + rpx], outline=_lerp(col, (255, 255, 255), 0.25), width=max(1, ss))
        d.ellipse([px - rpx * 0.45 - rpx * 0.25, py - rpx * 0.45 - rpx * 0.25,
                   px - rpx * 0.45 + rpx * 0.25, py - rpx * 0.45 + rpx * 0.25],
                  fill=_lerp(col, (255, 255, 255), 0.55))

    # near (own) goal is closest to the camera -> draw it AFTER the bodies so the
    # net sits IN FRONT of the player. Correct for both views because near_x is
    # computed per-camera (x=0 for agent 0, x=width for agent 1).
    draw_net_goal(near_x, (210, 170, 90), (140, 110, 60, 180), near_out)  # own (amber)

    return im.resize((out_px, out_px), Image.LANCZOS)


def rollout(cfg, seed):
    o = ArenaOracle(cfg, seed=seed)
    s = policy_strike_to_goal(o, 0)
    d = policy_defend_goal(o, 1, side="right") if cfg.n_mallets == 2 else None
    states = [ArenaOracle.state.__get__(o)()] if False else None
    snaps = [_snapshot(o)]
    while not o.done:
        acts = [s()] + ([d()] if d else [])
        o.step(acts)
        snaps.append(_snapshot(o))
    return snaps, o.outcome


class _Frozen:
    """A tiny frozen view of oracle state so we can re-render both cams per frame."""
    def __init__(self, puck, mallets, cfg, goal):
        self.puck = puck; self.mallets = mallets; self.cfg = cfg; self.goal = goal


class _Body:
    def __init__(self, x, y): self.position = _P(x, y)


class _P:
    def __init__(self, x, y): self.x = x; self.y = y


def _snapshot(o):
    puck = _Body(o.puck.position.x, o.puck.position.y)
    mallets = [_Body(m.position.x, m.position.y) for m in o.mallets]
    return _Frozen(puck, mallets, o.cfg, o.goal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent", type=int, default=0)
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--mallets", type=int, default=2)
    ap.add_argument("--out", type=str, default="smooth_angled.gif")
    ap.add_argument("--px", type=int, default=240)
    ap.add_argument("--tilt", type=float, default=40.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    cfg = ArenaConfig(mode="airhockey", n_mallets=args.mallets)
    snaps, outcome = rollout(cfg, args.seed)

    frames = []
    for snap in snaps:
        if args.both:
            a = render_smooth(snap, 0, out_px=args.px, tilt_deg=args.tilt)
            b = render_smooth(snap, 1, out_px=args.px, tilt_deg=args.tilt)
            gap = 10
            canvas = Image.new("RGB", (args.px * 2 + gap, args.px), (12, 12, 16))
            canvas.paste(a, (0, 0)); canvas.paste(b, (args.px + gap, 0))
            frames.append(canvas)
        else:
            frames.append(render_smooth(snap, args.agent, out_px=args.px, tilt_deg=args.tilt))

    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0)
    print(f"wrote {args.out} ({len(frames)} frames, outcome={outcome})")


if __name__ == "__main__":
    main()
