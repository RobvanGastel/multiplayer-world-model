from __future__ import annotations
import argparse

import numpy as np
from PIL import Image, ImageDraw

from wm.env.arena import ArenaConfig, ArenaOracle
from wm.env.policies import make_policies


PUCK_DRAW_FRAC = 1.0
MALLET_DRAW_FRAC = 1.0
PUCK_HEIGHT_FRAC = 0.25
MALLET_HEIGHT_FRAC = 0.3

class _Cam:
    def __init__(self, cfg, agent_idx, W, Hpx, tilt_deg=35.0):
        import math as _m
        self.cfg = cfg
        self.W = W
        self.H = Hpx
        # agent 0 sits behind the LEFT wall looking toward +x (far goal at
        # right); agent 1 sits behind the RIGHT wall looking the other way.
        self.flip = (agent_idx == 1)
        tilt = _m.radians(tilt_deg)
        comp = _m.cos(tilt) # vertical compression of depth
        span = 0.78 * comp
        # pull the near edge up from the very bottom so there's foreground
        # floor: the own goal (near) sits clearly in front of the player.
        top = 0.5 - span / 2 + 0.02
        self.horizon_y = Hpx * top
        self.near_y = Hpx * (top + span) # For the playing field

        self.ref_aspect = 72.0 / 120.0
        self.aspect_scale = (cfg.height / cfg.width) / self.ref_aspect
        self.near_hw = W * 0.42 * self.aspect_scale
        self.far_hw = W * (0.42 * (0.55 + 0.45 * comp)) * self.aspect_scale

    def depth(self, x):
        """0 at the viewer's near edge, 1 at the far edge."""
        d = x / self.cfg.width
        return (1.0 - d) if self.flip else d

    def project(self, x, y):
        t = self.depth(x) # perspective easing

        k = 0.15
        t_safe = max(-5.0, min(5.0, t))
        pt = t_safe * (1.0 + k) / (1.0 + k * t_safe)

        py = self.near_y + (self.horizon_y - self.near_y) * pt
        hw = self.near_hw + (self.far_hw - self.near_hw) * pt
        v = (y / self.cfg.height) - 0.5
        px = self.W * 0.5 + v * (2 * hw)
        return px, py, (2 * hw) / self.cfg.height

    def depth_scale(self, x, y, eps=0.5):
        """Local screen-pixels-per-world-unit in the DEPTH (x) direction at
        this point, i.e. |d(py)/dx|."""
        y_plus = self.project(x + eps, y)[1]
        y_minus = self.project(x - eps, y)[1]
        return abs(y_plus - y_minus) / (2 * eps)


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _goal_colors(board_x, cfg):
    """Post/net/crease colors for the goal at this board x, fixed by which
    mallet's goal it is."""

    # Returns the (post_colour, net_colour, crease_colour)
    if board_x < cfg.width / 2:
        return (90, 200, 130), (60, 120, 85, 180), (90, 200, 130, 200)
    return (210, 170, 90), (140, 110, 60, 180), (210, 170, 90, 200)


def _draw_net_goal(d, cam, ss, board_x, gy0, gy1, outward, post_col, net_col):
    """Upright goal frame on the goal line, net sloping back+down to a bar on
    the ground a half-tile behind it."""
    setback = (cam.cfg.width / 6.0) * 0.5
    back_x = board_x + outward * setback
    f0 = cam.project(board_x, gy0)[:2]
    f1 = cam.project(board_x, gy1)[:2]
    b0 = cam.project(back_x, gy0)[:2]
    b1 = cam.project(back_x, gy1)[:2]
    post_h = 9 * ss * cam.project(board_x, gy0)[2]

    t0 = (f0[0], f0[1] - post_h)
    t1 = (f1[0], f1[1] - post_h)

    # semi-transparent, not opaque: the near goal draws its net AFTER bodies
    d.polygon([f0, f1, b1, b0], fill=(28, 32, 44, 190))
    d.polygon([t0, t1, b1, b0], fill=(net_col[0], net_col[1], net_col[2], 70))
    nn = 6
    for kk in range(nn + 1):
        fx = t0[0] + (t1[0] - t0[0]) * kk / nn
        fy = t0[1] + (t1[1] - t0[1]) * kk / nn
        bx = b0[0] + (b1[0] - b0[0]) * kk / nn
        by = b0[1] + (b1[1] - b0[1]) * kk / nn
        d.line([(fx, fy), (bx, by)], fill=net_col, width=max(1, ss))
    for r in (0.33, 0.66):
        ax = t0[0] + (b0[0] - t0[0]) * r; ay = t0[1] + (b0[1] - t0[1]) * r
        cx = t1[0] + (b1[0] - t1[0]) * r; cy = t1[1] + (b1[1] - t1[1]) * r
        d.line([(ax, ay), (cx, cy)], fill=net_col, width=max(1, ss))
    d.polygon([t0, f0, b0], fill=(net_col[0], net_col[1], net_col[2], 45))
    d.polygon([t1, f1, b1], fill=(net_col[0], net_col[1], net_col[2], 45))

    d.line([f0, t0], fill=post_col, width=max(2, int(ss * 2.6)))
    d.line([f1, t1], fill=post_col, width=max(2, int(ss * 2.6)))
    d.line([t0, t1], fill=post_col, width=max(2, int(ss * 2.6)))

    # back boundary of the pocket, in the goal's own color.
    d.line([b0, b1], fill=post_col, width=max(2, int(ss * 2.6)))
    d.line([t0, b0], fill=post_col, width=max(2, int(ss * 1.5)))
    d.line([t1, b1], fill=post_col, width=max(2, int(ss * 1.5)))


def _draw_goal_crease(d, cam, ss, goal_x, gy0, gy1, into_sign, color, n=24):
    """Painted half-circle in front of a goal"""
    gmid = (gy0 + gy1) / 2.0
    r = (gy1 - gy0) / 2.0
    pts = []
    for i in range(n + 1):
        theta = -np.pi / 2 + np.pi * i / n
        wx = goal_x + into_sign * r * np.cos(theta)
        wy = gmid + r * np.sin(theta)
        px, py, _ = cam.project(wx, wy)
        pts.append((px, py))
    d.line(pts, fill=color, width=max(1, ss))


def render_iso_pil(oracle: ArenaOracle, agent_idx: int = 0, out_px: int | None = None,
                    ss: int = 3, tilt_deg: float = 35.0) -> Image.Image:
    """Render one agent's 2.5D view as a PIL Image at out_px (defaults to
    cfg.frame_size)."""
    cfg = oracle.cfg
    out_px = out_px or cfg.frame_size
    W = H = out_px * ss
    cam = _Cam(cfg, agent_idx, W, H, tilt_deg=tilt_deg)
    im = Image.new("RGB", (W, H), (16, 18, 26))
    d = ImageDraw.Draw(im, "RGBA")

    gy0, gy1 = oracle.goal["y0"], oracle.goal["y1"]

    # floor trapezoid
    c = [cam.project(0, 0), cam.project(0, cfg.height),
         cam.project(cfg.width, cfg.height), cam.project(cfg.width, 0)]
    d.polygon([(p[0], p[1]) for p in c], fill=(34, 39, 54))

    # even grid on the floor
    grid = (48, 54, 72)
    for gx in range(0, int(cfg.width) + 1, max(1, int(cfg.width // 6))):
        a = cam.project(gx, 0); b = cam.project(gx, cfg.height)
        d.line([(a[0], a[1]), (b[0], b[1])], fill=grid, width=max(1, ss))
    for gy in range(0, int(cfg.height) + 1, max(1, int(cfg.height // 4))):
        a = cam.project(0, gy); b = cam.project(cfg.width, gy)
        d.line([(a[0], a[1]), (b[0], b[1])], fill=grid, width=max(1, ss))

    # center line
    m0 = cam.project(cfg.width * 0.5, 0); m1 = cam.project(cfg.width * 0.5, cfg.height)
    d.line([(m0[0], m0[1]), (m1[0], m1[1])], fill=(90, 98, 124), width=max(2, ss * 2))

    far_x = cfg.width if not cam.flip else 0.0
    near_x = 0.0 if not cam.flip else cfg.width
    far_out = +1 if not cam.flip else -1
    near_out = -1 if not cam.flip else +1
    # far/near only control draw order and outward direction (which camera
    # is looking at which side).
    far_post, far_net, far_crease = _goal_colors(far_x, cfg)
    near_post, near_net, near_crease = _goal_colors(near_x, cfg)

    _draw_goal_crease(d, cam, ss, far_x, gy0, gy1, -far_out, far_crease)
    _draw_goal_crease(d, cam, ss, near_x, gy0, gy1, -near_out, near_crease)
    # far (opponent) goal is deepest -> draw before the bodies
    _draw_net_goal(d, cam, ss, far_x, gy0, gy1, far_out, far_post, far_net)

    # bodies far -> near with soft shadows and a highlight
    bodies = [("puck", oracle.puck.position, (240, 220, 90), cfg.puck_radius,
               PUCK_DRAW_FRAC, PUCK_HEIGHT_FRAC)]
    mcol = [(90, 160, 240), (240, 110, 110)]
    for i, b in enumerate(oracle.mallets):
        bodies.append(("mallet", b.position, mcol[i % 2], cfg.mallet_radius,
                        MALLET_DRAW_FRAC, MALLET_HEIGHT_FRAC))
    bodies.sort(key=lambda t: cam.depth(t[1].x), reverse=True)

    for kind, pos, col, true_radius, draw_frac, height_frac in bodies:
        # Anisotropic to shape the puck and mallets properly.
        px, py, sc = cam.project(pos.x, pos.y)
        depth_sc = cam.depth_scale(pos.x, pos.y)
        rpx_lat = true_radius * draw_frac * sc
        rpx_depth = true_radius * draw_frac * depth_sc

        # height is a third, purely cosmetic screen-up axis.
        in_pocket = pos.x < 0 or pos.x > cfg.width
        height_px = 0.0 if in_pocket else true_radius * height_frac * sc
        top_py = py - height_px

        # shadow stays at floor level (py), not the raised top
        d.ellipse([px - rpx_lat * 1.15, py + rpx_depth * 0.15,
                   px + rpx_lat * 1.15, py + rpx_depth * 0.75],
                  fill=(0, 0, 0, 110))
        # cylindrical side: a "capsule" (rounded top AND bottom, straight
        # sides between) rather than a plain rectangle.
        side_col = _lerp(col, (0, 0, 0), 0.35)
        d.ellipse([px - rpx_lat, py - rpx_depth, px + rpx_lat, py + rpx_depth], fill=side_col)
        d.rectangle([px - rpx_lat, top_py, px + rpx_lat, py], fill=side_col)
        # top cap: fill + rim + highlight, same look as the old flat disc,
        # just drawn height_px higher
        d.ellipse([px - rpx_lat, top_py - rpx_depth, px + rpx_lat, top_py + rpx_depth],
                  fill=col)
        d.ellipse([px - rpx_lat, top_py - rpx_depth, px + rpx_lat, top_py + rpx_depth],
                  outline=_lerp(col, (255, 255, 255), 0.25), width=max(1, ss))
        d.ellipse([px - rpx_lat * 0.7, top_py - rpx_depth * 0.7,
                   px - rpx_lat * 0.2, top_py - rpx_depth * 0.2],
                  fill=_lerp(col, (255, 255, 255), 0.55))

    # near (own) goal drawn AFTER the bodies so the net sits in front of the player
    _draw_net_goal(d, cam, ss, near_x, gy0, gy1, near_out, near_post, near_net)

    return im.resize((out_px, out_px), Image.LANCZOS)


def render_iso(oracle: ArenaOracle, agent_idx: int = 0, out_px: int | None = None,
               ss: int = 3, tilt_deg: float = 35.0) -> np.ndarray:
    """Same view as render_iso_pil, as an (S,S,3) uint8 array — the format used
    for dataset frames alongside render_flat()."""
    im = render_iso_pil(oracle, agent_idx=agent_idx, out_px=out_px, ss=ss, tilt_deg=tilt_deg)
    return np.array(im.convert("RGB"), dtype=np.uint8)
