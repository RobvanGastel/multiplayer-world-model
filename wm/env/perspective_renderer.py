"""
'angled' renderer — a fixed 45-degree perspective view from behind one agent's
goal, looking down the board toward the far goal. Render-only; physics untouched.

The camera sits behind the chosen agent's goal, tilted down ~45 deg. The board
recedes away from the viewer: the near edge (your goal) is wide and low in frame,
the far edge (opponent goal) is narrow and high. Bodies are scaled by their
depth down-field, so the puck visibly travels away from you toward the far goal.

This is the same "project the top-down state" trick as the table/iso renderers,
just with a perspective transform. No occlusion, fixed camera, action space
unchanged — so the world model's job stays tractable while the view gains real
depth and becomes egocentric (per-agent), which reads toward embodied/robotics.

Because it's a pure function of oracle state, you can render EITHER top-down or
angled frames from the same simulation and compare world models trained on each.

Usage:
    python perspective_renderer.py --preview-gif angled.gif
    python perspective_renderer.py --compare-gif ang_compare.gif       # topdown | angled
    python perspective_renderer.py --agent 0 --preview-gif a0.gif      # which goal to sit behind
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
# perspective projection
# --------------------------------------------------------------------------
# We map board coords (x in [0,width], y in [0,height]) to screen.
# "Depth" runs along x: the agent behind the LEFT goal looks toward +x (far goal
# on the right). We reorient so depth increases INTO the screen regardless of
# which agent's view we render.

class Camera:
    def __init__(self, cfg: ArenaConfig, agent_idx: int, S: int):
        self.cfg = cfg
        self.S = S
        # agent 0 sits behind the LEFT wall looking toward +x (far goal at right).
        # agent 1 sits behind the RIGHT wall looking toward -x (mirror).
        self.flip = (agent_idx == 1)
        # horizon / vanishing placement in frame. Tuned so the board's depth
        # midpoint (where the center line sits) projects to the frame's vertical
        # center — otherwise the midline reads as sitting too low.
        self.horizon_y = S * 0.19     # far edge
        self.near_y = S * 0.81        # near edge (viewer's goal lip)
        self.far_halfw = S * 0.27
        self.near_halfw = S * 0.46

    def depth(self, x):
        """0 at the viewer's near edge, 1 at the far edge."""
        d = x / self.cfg.width
        return (1.0 - d) if self.flip else d

    def project(self, x, y):
        """board (x,y) -> (screen_px, screen_py, scale). depth 0=near .. 1=far."""
        S = self.S
        t = self.depth(x)                       # 0 near .. 1 far
        pt = t / (1.0 + 0.15 * (1.0 - t))       # gentle easing; midpoint ~ center
        py = self.near_y + (self.horizon_y - self.near_y) * pt
        halfw = self.near_halfw + (self.far_halfw - self.near_halfw) * pt
        # lateral position across the board width
        v = (y / self.cfg.height) - 0.5         # -0.5..0.5
        px = S * 0.5 + v * (2 * halfw)
        # scale for body sizes: near big, far small
        scale = halfw / self.near_halfw
        return px, py, scale


def _disc(img, cx, cy, r, color, alpha=1.0):
    S = img.shape[0]
    r = max(0.8, r)
    x0, x1 = max(0, int(cx - r - 1)), min(S, int(cx + r + 2))
    y0, y1 = max(0, int(cy - r - 1)), min(S, int(cy + r + 2))
    col = np.array(color, np.float32)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            d = math.hypot(xx - cx, yy - cy)
            if d <= r:
                a = alpha if (r - d) >= 1 else alpha * max(0.0, r - d)
                img[yy, xx] = (1 - a) * img[yy, xx] + a * col


def _ellipse_shadow(img, cx, cy, rx, ry, strength):
    S = img.shape[0]
    x0, x1 = max(0, int(cx - rx - 1)), min(S, int(cx + rx + 2))
    y0, y1 = max(0, int(cy - ry - 1)), min(S, int(cy + ry + 2))
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            dx = (xx - cx) / (rx + 1e-6); dy = (yy - cy) / (ry + 1e-6)
            dd = dx * dx + dy * dy
            if dd <= 1.0:
                a = strength * (1 - dd) ** 1.4
                img[yy, xx] = (1 - a) * img[yy, xx]


MALLET_COLORS = [(90, 160, 240), (240, 110, 110)]
MALLET_HI = [(200, 225, 255), (255, 200, 200)]


def _fill_quad(img, pts, color):
    S = img.shape[0]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = max(0, int(min(xs))), min(S, int(max(xs)) + 1)
    y0, y1 = max(0, int(min(ys))), min(S, int(max(ys)) + 1)
    poly = pts
    col = np.array(color, np.float32)
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            c = False; n = len(poly); j = n - 1
            for i in range(n):
                xi, yi = poly[i]; xj, yj = poly[j]
                if ((yi > yy) != (yj > yy)) and \
                   (xx < (xj - xi) * (yy - yi) / (yj - yi + 1e-9) + xi):
                    c = not c
                j = i
            if c:
                img[yy, xx] = col


def render_angled(oracle: ArenaOracle, agent_idx: int = 0) -> np.ndarray:
    cfg = oracle.cfg
    S = cfg.frame_size
    cam = Camera(cfg, agent_idx, S)
    img = np.zeros((S, S, 3), np.float32)

    # sky / back wall above the horizon, floor below
    img[:] = (20, 23, 33)
    # floor quad (perspective trapezoid)
    c0 = cam.project(0, 0); c1 = cam.project(0, cfg.height)
    c2 = cam.project(cfg.width, cfg.height); c3 = cam.project(cfg.width, 0)
    if cam.flip:
        c0 = cam.project(cfg.width, 0); c1 = cam.project(cfg.width, cfg.height)
        c2 = cam.project(0, cfg.height); c3 = cam.project(0, 0)
    floor = [(c0[0], c0[1]), (c1[0], c1[1]), (c2[0], c2[1]), (c3[0], c3[1])]
    _fill_quad(img, floor, (36, 41, 56))

    # center line (draw as a projected segment across the board mid-x)
    midx = cfg.width * 0.5
    a = cam.project(midx, 0); b = cam.project(midx, cfg.height)
    steps = 24
    for i in range(steps + 1):
        yy = cfg.height * i / steps
        px, py, _ = cam.project(midx, yy)
        if 0 <= int(py) < S and 0 <= int(px) < S:
            img[int(py), int(px)] = img[int(py), int(px)] * 0.6 + np.array((64, 72, 92)) * 0.4

    # goal mouths on BOTH ends of the board.
    # far end = opponent's goal (narrow, near horizon); near end = viewer's own
    # goal (wide, at the bottom lip). Both span the goal's y-range.
    g = oracle.goal
    far_x = cfg.width if not cam.flip else 0.0
    near_x = 0.0 if not cam.flip else cfg.width

    def draw_goal_band(board_x, color_top, color_face):
        a = cam.project(board_x, g["y0"])
        b = cam.project(board_x, g["y1"])
        # a thin band with a slightly darker face below it for a lipped look
        _fill_quad(img, [
            (a[0], a[1] - 3), (b[0], b[1] - 3),
            (b[0], b[1] + 1), (a[0], a[1] + 1)
        ], color_top)
        _fill_quad(img, [
            (a[0], a[1] + 1), (b[0], b[1] + 1),
            (b[0], b[1] + 4), (a[0], a[1] + 4)
        ], color_face)

    # opponent goal (far) — the target you're shooting at, brighter green
    draw_goal_band(far_x, (70, 150, 100), (40, 90, 62))
    # your own goal (near) — the one you defend, cooler/dimmer so it reads as "yours"
    draw_goal_band(near_x, (150, 120, 70), (92, 74, 42))

    # collect bodies with depth for painter's-algorithm ordering (far first)
    bodies = []
    p = oracle.puck.position
    pv = oracle.puck.velocity
    pspeed = math.hypot(pv.x, pv.y)
    bodies.append(("puck", p.x, p.y, cfg.puck_radius, (240, 220, 90), None, pspeed))
    for i, bod in enumerate(oracle.mallets):
        bodies.append(("mallet", bod.position.x, bod.position.y, cfg.mallet_radius,
                       MALLET_COLORS[i % 2], MALLET_HI[i % 2], 0.0))
    # sort far -> near (draw far first)
    bodies.sort(key=lambda t: cam.depth(t[1]), reverse=True)

    for kind, bx, by, br, col, hi, spd in bodies:
        px, py, scale = cam.project(bx, by)
        # radius must match the projection's LOCAL lateral pixel density, else
        # side-by-side bodies show a gap while touching. lateral density =
        # screen-pixels per world-unit across the board width at this depth.
        halfw = cam.near_halfw + (cam.far_halfw - cam.near_halfw) * (
            cam.depth(bx) / (1.0 + 0.15 * (1.0 - cam.depth(bx))))
        lateral_px_per_unit = (2 * halfw) / cfg.height
        rpx = br * lateral_px_per_unit * 1.15
        # hop for puck by speed (render-only), shadow stays on floor
        hop = 0.0
        if kind == "puck":
            hop = min(1.0, spd / cfg.puck_max_speed) * (S * 0.05) * scale
        _ellipse_shadow(img, px, py + rpx * 0.25, rpx * 1.2, rpx * 0.5,
                        strength=0.5)
        _disc(img, px, py - hop, rpx, col)
        if hi is not None:
            _disc(img, px - rpx * 0.3, py - hop - rpx * 0.3, rpx * 0.35, hi)
        else:
            _disc(img, px - rpx * 0.3, py - hop - rpx * 0.3, rpx * 0.3, (255, 248, 200))

    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# rollout + preview
# --------------------------------------------------------------------------

def rollout_frames(cfg, seed, render_fn, policies="strike"):
    o = ArenaOracle(cfg, seed=seed)
    if policies == "random":
        pols = [policy_random(o, i) for i in range(cfg.n_mallets)]
    elif cfg.mode == "pursuit" and cfg.n_mallets >= 2:
        pols = [policy_chase_puck(o, 0), policy_evade(o, 1)]
    else:
        pols = [policy_strike_to_goal(o, i) for i in range(cfg.n_mallets)]
    frames = [render_fn(o)]
    while not o.done:
        o.step([pl() for pl in pols])
        frames.append(render_fn(o))
    return frames, o.outcome


def save_gif(frames, path, scale=5, fps=30):
    imgs = [Image.fromarray(f, "RGB").resize(
        (f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)


def save_compare(cfg, seed, path, agent_idx=0, scale=5, fps=30):
    top, _ = rollout_frames(cfg, seed, lambda o: o._render_flat())
    ang, _ = rollout_frames(cfg, seed, lambda o: render_angled(o, agent_idx))
    n = min(len(top), len(ang)); gap = 6
    imgs = []
    for i in range(n):
        S = top[i].shape[0]
        canvas = np.full((S, S * 2 + gap, 3), 12, np.uint8)
        canvas[:, :S] = top[i]; canvas[:, S + gap:] = ang[i]
        imgs.append(Image.fromarray(canvas, "RGB").resize(
            ((S * 2 + gap) * scale, S * scale), Image.NEAREST))
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview-gif", type=str, default="")
    ap.add_argument("--compare-gif", type=str, default="")
    ap.add_argument("--agent", type=int, default=0, help="which goal to sit behind (0 or 1)")
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
        save_compare(cfg, args.seed, args.compare_gif, agent_idx=args.agent)
        print(f"wrote {args.compare_gif} (left=topdown, right=angled agent {args.agent})")
    if args.preview_gif:
        frames, outcome = rollout_frames(cfg, args.seed, lambda o: render_angled(o, args.agent))
        save_gif(frames, args.preview_gif)
        print(f"wrote {args.preview_gif} ({len(frames)} frames, outcome={outcome})")
    if not args.preview_gif and not args.compare_gif:
        print("pass --preview-gif or --compare-gif")


if __name__ == "__main__":
    main()
