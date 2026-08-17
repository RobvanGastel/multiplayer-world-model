"""
Air-hockey arena oracle simulator.

This is the *oracle* half of a MIRA-style world-model pipeline: a deterministic
Pymunk simulator that produces ground-truth trajectories. You train a pixel/
latent-frame world model to imitate it, and at play time the model — not this
sim — generates the world. This file is the data source and the evaluation
oracle, never the thing the player touches once the model is trained.

Design decisions (all chosen to keep the plan finishable and multiplayer-reachable):

  * AGENT-LIST FIRST. The sim holds a *list* of mallets. Run it with one for the
    single-agent world model; flip to two for head-to-head or pursuit later with
    no physics change. Multiplayer is a flag, not a rewrite.

  * NEUTRAL PUCK is the star. The mallet dynamics are simple; the puck caroming
    off walls and mallets is the rich, skill-expressive, hard-to-predict part the
    world model actually has to learn. Present even in single-agent time-trial.

  * HIDDEN FORCE FIELD. An optional latent drift on the puck. OFF by default so
    the pipeline works cleanly first; turn ON to give the model real hidden state
    to infer from motion alone (the interesting representation-learning problem).

  * PLUGGABLE RENDERER. 'flat' top-down (easiest for the world model) and 'iso'
    (2.5D look) share one physics core and one state dump. Train on flat, flip to
    iso for the 2.5D version with zero physics changes.

  * DUMPS BOTH pixels (to train on) and states (ground-truth to evaluate against).

Modes: 'airhockey' (knock puck into goal) and 'pursuit' (arcade tag) share ~90%
of the code and differ only in objective/terminal condition.

Requires: pymunk, numpy, pillow
"""

from __future__ import annotations
import math
import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import pymunk
from PIL import Image


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class ArenaConfig:
    # --- overall size scale. Everything spatial scales together so the game
    # feels identical, just roomier. 1.0 = original 120x72 board. ---
    scale: float = 3.0

    width: float = 120.0
    height: float = 72.0
    wall_thickness: float = 3.0

    mallet_radius: float = 8.0
    mallet_mass: float = 3.0
    puck_radius: float = 6.0
    puck_mass: float = 1.0

    # dynamics feel
    puck_elasticity: float = 0.88      # lively banks, still controllable
    wall_elasticity: float = 0.98
    mallet_elasticity: float = 0.85
    puck_friction: float = 0.02        # near-frictionless plane
    linear_damping: float = 0.15       # gentle global drag so energy bleeds off

    thrust: float = 900.0              # force magnitude per action step
    mallet_max_speed: float = 220.0
    puck_max_speed: float = 320.0

    # hidden latent: a constant force field on the puck, off by default
    hidden_field: bool = False
    hidden_field_max: float = 120.0    # sampled magnitude when enabled

    dt: float = 1.0 / 60.0
    substeps: int = 2                  # physics substeps per env step

    frame_size: int = 64
    renderer: str = "flat"             # 'flat' or 'iso'
    mode: str = "airhockey"            # 'airhockey' or 'pursuit'
    n_mallets: int = 1                 # 1 = single-agent; 2 = head-to-head/pursuit

    max_steps: int = 400

    def __post_init__(self):
        # Apply the overall size scale so the board is bigger but the game feels
        # the same. Lengths scale by s. To keep acceleration and time-to-cross
        # (in seconds) identical, velocities scale by s and forces by s too
        # (a = F/m must scale by s to match the s-scaled distances over the same
        # dt). Masses, elasticities, damping, dt are dimensionless-in-feel and
        # stay put. Frame size is display-only and is left alone.
        s = self.scale
        if s == 1.0:
            return
        self.width *= s
        self.height *= s
        self.wall_thickness *= s
        self.mallet_radius *= s
        self.puck_radius *= s
        self.thrust *= s
        self.mallet_max_speed *= s
        self.puck_max_speed *= s
        self.hidden_field_max *= s


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

class ArenaOracle:
    """Deterministic air-hockey / pursuit arena. Seed it for reproducible trajectories."""

    def __init__(self, cfg: ArenaConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._build()

    # ---- construction -----------------------------------------------------

    def _build(self):
        cfg = self.cfg
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 1.0 - cfg.linear_damping  # pymunk damping is a retain factor

        self._add_walls()
        self.mallets = [self._add_mallet(i) for i in range(cfg.n_mallets)]
        self.puck = self._add_puck()

        # hidden latent force field (sampled once per episode, applied to puck)
        if cfg.hidden_field:
            ang = self.rng.uniform(0, 2 * math.pi)
            mag = self.rng.uniform(0.3, 1.0) * cfg.hidden_field_max
            self.field = (math.cos(ang) * mag, math.sin(ang) * mag)
        else:
            self.field = (0.0, 0.0)

        # goal region for airhockey: the goal IS the wall line. Scoring requires
        # the puck's leading edge to actually reach the wall (to the pixel), not
        # just its center to enter a fat zone. Checked in _check_terminal.
        gh = cfg.height * 0.5
        self.goal = dict(
            x0=cfg.width, x1=cfg.width,   # the right wall line
            y0=(cfg.height - gh) / 2, y1=(cfg.height + gh) / 2,
        )
        self.steps = 0
        self.done = False
        self.outcome = None  # 'goal' | 'caught' | 'timeout'

    def _add_walls(self):
        cfg = self.cfg
        t = cfg.wall_thickness
        w, h = cfg.width, cfg.height
        gh = h * 0.5
        gy0, gy1 = (h - gh) / 2, (h + gh) / 2   # goal slot y-range

        # top and bottom are solid
        segs = [((0, 0), (w, 0)), ((0, h), (w, h))]
        # left and right walls have a GAP at the goal slot so the puck can enter
        # and score to the pixel instead of bouncing off the goal.
        segs += [((0, 0), (0, gy0)), ((0, gy1), (0, h))]     # left wall, gapped
        segs += [((w, 0), (w, gy0)), ((w, gy1), (w, h))]     # right wall, gapped

        for a, b in segs:
            seg = pymunk.Segment(self.space.static_body, a, b, t / 2)
            seg.elasticity = cfg.wall_elasticity
            seg.friction = 0.1
            seg.collision_type = 1
            self.space.add(seg)

    def _add_mallet(self, idx: int):
        cfg = self.cfg
        m = pymunk.moment_for_circle(cfg.mallet_mass, 0, cfg.mallet_radius)
        body = pymunk.Body(cfg.mallet_mass, m)
        # In two-mallet airhockey, mallet 0 = striker (left, attacks far goal),
        # mallet 1 = defender (right, guards far goal). Otherwise spread on the left.
        if cfg.mode == "airhockey" and cfg.n_mallets == 2:
            if idx == 0:
                body.position = (cfg.width * 0.2, cfg.height * 0.5)
            else:
                body.position = (cfg.width * 0.82, cfg.height * 0.5)
        else:
            frac = (idx + 1) / (cfg.n_mallets + 1)
            body.position = (cfg.width * 0.2, cfg.height * frac)
        shape = pymunk.Circle(body, cfg.mallet_radius)
        shape.elasticity = cfg.mallet_elasticity
        shape.friction = 0.2
        shape.collision_type = 2
        self.space.add(body, shape)
        return body

    def _add_puck(self):
        cfg = self.cfg
        m = pymunk.moment_for_circle(cfg.puck_mass, 0, cfg.puck_radius)
        body = pymunk.Body(cfg.puck_mass, m)
        body.position = (cfg.width * 0.5, cfg.height * 0.5)
        # small random initial nudge so episodes differ
        ang = self.rng.uniform(0, 2 * math.pi)
        speed = self.rng.uniform(20, 60)
        body.velocity = (math.cos(ang) * speed, math.sin(ang) * speed)
        shape = pymunk.Circle(body, cfg.puck_radius)
        shape.elasticity = cfg.puck_elasticity
        shape.friction = cfg.puck_friction
        shape.collision_type = 3
        self.space.add(body, shape)
        return body

    # ---- stepping ---------------------------------------------------------

    def _clamp_speed(self, body, max_speed):
        v = body.velocity
        s = math.hypot(v.x, v.y)
        if s > max_speed:
            body.velocity = (v.x / s * max_speed, v.y / s * max_speed)

    def step(self, actions):
        """actions: list of (ax, ay) in [-1,1] per mallet. Returns (frame, state, done)."""
        cfg = self.cfg
        if len(actions) != len(self.mallets):
            raise ValueError(f"expected {len(self.mallets)} actions, got {len(actions)}")

        for _ in range(cfg.substeps):
            for body, (ax, ay) in zip(self.mallets, actions):
                ax = max(-1.0, min(1.0, ax))
                ay = max(-1.0, min(1.0, ay))
                body.apply_force_at_local_point((ax * cfg.thrust, ay * cfg.thrust), (0, 0))
            # hidden field pushes the puck
            if self.field != (0.0, 0.0):
                self.puck.apply_force_at_local_point(self.field, (0, 0))
            self.space.step(cfg.dt / cfg.substeps)
            for body in self.mallets:
                self._clamp_speed(body, cfg.mallet_max_speed)
            self._clamp_speed(self.puck, cfg.puck_max_speed)
            self._confine_mallets()

        return self._post_step()

    def _confine_mallets(self):
        """Keep mallets inside the playfield and, in 2-mallet airhockey, each in
        its own half. Mallets never leave through the goal gaps — those are for
        the puck only."""
        cfg = self.cfg
        r = cfg.mallet_radius
        w, h = cfg.width, cfg.height

        # (1) hard-clamp every mallet inside the board bounds
        for m in self.mallets:
            x, y = m.position.x, m.position.y
            vx, vy = m.velocity.x, m.velocity.y
            if x < r:      x = r;      vx = max(0.0, vx)
            if x > w - r:  x = w - r;  vx = min(0.0, vx)
            if y < r:      y = r;      vy = max(0.0, vy)
            if y > h - r:  y = h - r;  vy = min(0.0, vy)
            m.position = (x, y)
            m.velocity = (vx, vy)

        # (2) center-line rule for the 2-mallet game
        if not (cfg.mode == "airhockey" and cfg.n_mallets == 2):
            return
        mid = cfg.width * 0.5
        s = self.mallets[0]
        if s.position.x > mid - r:
            s.position = (mid - r, s.position.y)
            if s.velocity.x > 0:
                s.velocity = (0.0, s.velocity.y)
        d = self.mallets[1]
        if d.position.x < mid + r:
            d.position = (mid + r, d.position.y)
            if d.velocity.x < 0:
                d.velocity = (0.0, d.velocity.y)

    def _post_step(self):
        self.steps += 1
        self._check_terminal()
        return self.render(), self.state(), self.done

    def _check_terminal(self):
        cfg = self.cfg
        if cfg.mode == "airhockey":
            p = self.puck.position
            g = self.goal
            r = cfg.puck_radius
            # pixel-precise: the puck's leading edge must actually reach the goal
            # line (the wall), and its center must be within the goal's y-slot.
            in_slot = g["y0"] <= p.y <= g["y1"]
            reached_right = (p.x + r) >= cfg.width - 1e-6
            reached_left = (p.x - r) <= 0.0 + 1e-6
            if in_slot and reached_right:
                self.done = True
                self.outcome = "goal"          # striker scores far goal
            elif in_slot and reached_left and cfg.n_mallets == 2:
                self.done = True
                self.outcome = "conceded"      # own goal (2-mallet game)
        elif cfg.mode == "pursuit":
            # mallet 0 chases mallet 1 (needs n_mallets >= 2)
            if len(self.mallets) >= 2:
                a, b = self.mallets[0].position, self.mallets[1].position
                if math.hypot(a.x - b.x, a.y - b.y) < cfg.mallet_radius * 2.2:
                    self.done = True
                    self.outcome = "caught"
        if self.steps >= cfg.max_steps and not self.done:
            self.done = True
            self.outcome = "timeout"

    # ---- observation ------------------------------------------------------

    def state(self) -> dict:
        """Ground-truth low-dim state — for evaluation, not fed to a pixel model."""
        s = {
            "puck": [self.puck.position.x, self.puck.position.y,
                     self.puck.velocity.x, self.puck.velocity.y],
            "mallets": [[b.position.x, b.position.y, b.velocity.x, b.velocity.y]
                        for b in self.mallets],
            "field": list(self.field),
            "step": self.steps,
        }
        return s

    def render(self) -> np.ndarray:
        """Return an (S,S,3) uint8 frame using the configured renderer."""
        if self.cfg.renderer == "iso":
            return self._render_iso()
        return self._render_flat()

    def _canvas(self):
        S = self.cfg.frame_size
        return np.zeros((S, S, 3), dtype=np.uint8)

    def _disc(self, img, cx, cy, r, color):
        S = img.shape[0]
        r = max(1.0, r)
        x0, x1 = max(0, int(cx - r)), min(S, int(cx + r + 1))
        y0, y1 = max(0, int(cy - r)), min(S, int(cy + r + 1))
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r:
                    img[yy, xx] = color

    def _render_flat(self):
        cfg = self.cfg
        S = cfg.frame_size
        img = self._canvas()
        img[:] = (12, 14, 20)
        # uniform scale (letterbox) so circles + positions share one scale ->
        # contact renders true; center the board in the square frame
        sc = min(S / cfg.width, S / cfg.height)
        ox = (S - cfg.width * sc) / 2
        oy = (S - cfg.height * sc) / 2

        def X(wx): return ox + wx * sc
        def Y(wy): return oy + wy * sc

        # playfield panel
        x0, y0b = int(X(0)), int(Y(0))
        x1, y1b = int(X(cfg.width)), int(Y(cfg.height))
        img[y0b:y1b, x0:x1] = (26, 30, 42)

        if cfg.mode == "airhockey":
            g = self.goal
            gy0, gy1 = int(Y(g["y0"])), int(Y(g["y1"]))
            # center line exactly at board middle
            midx = int(X(cfg.width * 0.5))
            img[y0b:y1b, midx:midx + 1] = (60, 66, 86)
            # both goals
            img[gy0:gy1, x1 - 2:x1] = (70, 150, 100)     # right/opponent
            img[gy0:gy1, x0:x0 + 2] = (150, 120, 70)     # left/own

        p = self.puck.position
        self._disc(img, X(p.x), Y(p.y), cfg.puck_radius * sc, (240, 220, 90))
        mcolors = [(90, 160, 240), (240, 110, 110)]
        for i, b in enumerate(self.mallets):
            self._disc(img, X(b.position.x), Y(b.position.y),
                       cfg.mallet_radius * sc, mcolors[i % len(mcolors)])
        return img

    def _iso_project(self, x, y, S, cfg):
        # simple fixed isometric: rotate the plane and squash y, then center.
        nx = (x / cfg.width) - 0.5
        ny = (y / cfg.height) - 0.5
        ix = (nx - ny)
        iy = (nx + ny)
        px = S * (0.5 + ix * 0.42)
        py = S * (0.42 + iy * 0.30)
        return px, py

    def _render_iso(self):
        cfg = self.cfg
        S = cfg.frame_size
        img = self._canvas()
        img[:] = (14, 16, 22)

        # draw the floor quad corners for a sense of depth
        corners = [(0, 0), (cfg.width, 0), (cfg.width, cfg.height), (0, cfg.height)]
        proj = [self._iso_project(x, y, S, cfg) for x, y in corners]
        self._fill_quad(img, proj, (30, 33, 44))

        # shadows first (offset down), then bodies — gives the 2.5D read
        p = self.puck.position
        px, py = self._iso_project(p.x, p.y, S, cfg)
        self._disc(img, px, py + 2, cfg.puck_radius * 0.5, (8, 9, 12))
        self._disc(img, px, py, cfg.puck_radius * 0.55, (240, 220, 90))

        mcolors = [(90, 160, 240), (240, 110, 110)]
        for i, b in enumerate(self.mallets):
            bx, by = self._iso_project(b.position.x, b.position.y, S, cfg)
            self._disc(img, bx, by + 2, cfg.mallet_radius * 0.5, (8, 9, 12))
            self._disc(img, bx, by, cfg.mallet_radius * 0.6, mcolors[i % len(mcolors)])
        return img

    def _fill_quad(self, img, pts, color):
        S = img.shape[0]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = max(0, int(min(xs))), min(S, int(max(xs)) + 1)
        y0, y1 = max(0, int(min(ys))), min(S, int(max(ys)) + 1)
        poly = pts

        def inside(px, py):
            c = False
            n = len(poly)
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > py) != (yj > py)) and \
                   (px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi):
                    c = not c
                j = i
            return c

        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if inside(xx, yy):
                    img[yy, xx] = color


# --------------------------------------------------------------------------
# Scripted policies (to generate diverse, skillful data without a human)
# --------------------------------------------------------------------------

def policy_defend_goal(oracle: ArenaOracle, idx: int = 1, side: str = "right", noise=0.2):
    """Guard a goal: hold a defensive line just in front of the goal, track the
    puck's y to stay between it and the goal, and clear the puck (drive into it,
    away from the goal) when it gets close. This turns uncontested goals into
    blocked shots, rebounds, and clears — a much richer training distribution.
    """
    cfg = oracle.cfg
    rng = oracle.rng
    goal_x = cfg.width if side == "right" else 0.0
    # defensive line sits a bit in front of the goal
    line_x = cfg.width * 0.82 if side == "right" else cfg.width * 0.18
    clear_dist = (cfg.mallet_radius + cfg.puck_radius) * 3.0

    def act():
        b = oracle.mallets[idx].position
        bv = oracle.mallets[idx].velocity
        p = oracle.puck.position
        pv = oracle.puck.velocity
        dist = math.hypot(p.x - b.x, p.y - b.y)

        puck_threatening = (abs(p.x - goal_x) < cfg.width * 0.45)
        if dist < clear_dist and puck_threatening:
            # clear: hit the puck away from goal
            gx = goal_x - p.x
            cx, cy = -gx, (p.y - b.y)
            n = math.hypot(cx, cy) + 1e-6
            ax, ay = cx / n, cy / n
        else:
            # hold the line and cover the goal MOUTH. Predict where the puck will
            # cross the defensive x-line and get there first.
            approaching = (pv.x > 20) if side == "right" else (pv.x < -20)
            if approaching and abs(pv.x) > 1e-3:
                # time for puck to reach the line, then its y at that time
                tcross = (line_x - p.x) / pv.x
                tcross = max(0.0, min(tcross, 1.5))
                ty = p.y + pv.y * tcross
            else:
                ty = p.y
            # clamp to the goal mouth so the defender always covers the opening
            ty = max(oracle.goal["y0"], min(oracle.goal["y1"], ty))
            tx = line_x
            dx, dy = tx - b.x, ty - b.y
            n = math.hypot(dx, dy) + 1e-6
            # push harder the further out of position (bang-bang-ish for speed)
            gain = 1.0 if n < cfg.mallet_radius else 1.0
            ax, ay = dx / n * gain, dy / n * gain
        ax += rng.normal(0, noise); ay += rng.normal(0, noise)
        return (ax, ay)
    return act


def policy_strike_to_goal(oracle: ArenaOracle, idx: int = 0, noise=0.25):
    """Position behind the puck relative to the goal, then strike through it.

    This is the policy that makes the DATA good: it drives the mallet to the far
    side of the puck (away from the goal), lines up the goal direction, and pushes
    through — producing actual hits, bank shots, and goals rather than aimless
    drift. A world model trained on this sees the interesting events at density.
    """
    cfg = oracle.cfg
    rng = oracle.rng
    goal = (cfg.width - cfg.wall_thickness, cfg.height * 0.5)

    def act():
        b = oracle.mallets[idx].position
        p = oracle.puck.position
        # unit vector from puck toward goal
        gx, gy = goal[0] - p.x, goal[1] - p.y
        gn = math.hypot(gx, gy) + 1e-6
        gux, guy = gx / gn, gy / gn
        # the strike point is just behind the puck on the goal line
        strike_x = p.x - gux * (cfg.puck_radius + cfg.mallet_radius)
        strike_y = p.y - guy * (cfg.puck_radius + cfg.mallet_radius)

        bx, by = strike_x - b.x, strike_y - b.y
        dist_to_strike = math.hypot(bx, by)

        if dist_to_strike > cfg.mallet_radius * 1.2:
            # get into position behind the puck
            dx, dy = bx, by
        else:
            # in position: drive through the puck toward the goal
            dx, dy = gux, guy
        n = math.hypot(dx, dy) + 1e-6
        ax, ay = dx / n, dy / n
        ax += rng.normal(0, noise); ay += rng.normal(0, noise)
        return (ax, ay)
    return act


def policy_chase_puck(oracle: ArenaOracle, idx: int = 0, noise=0.35):
    """Simple lead-chase — kept for pursuit mode where the target is another mallet."""
    rng = oracle.rng
    def act():
        b = oracle.mallets[idx].position
        if oracle.cfg.mode == "pursuit" and len(oracle.mallets) >= 2:
            other = 1 - idx if idx < 2 else 0
            t = oracle.mallets[other].position
            tv = oracle.mallets[other].velocity
        else:
            t = oracle.puck.position
            tv = oracle.puck.velocity
        tx, ty = t.x + tv.x * 0.15, t.y + tv.y * 0.15
        dx, dy = tx - b.x, ty - b.y
        n = math.hypot(dx, dy) + 1e-6
        ax, ay = dx / n, dy / n
        ax += rng.normal(0, noise); ay += rng.normal(0, noise)
        return (ax, ay)
    return act


def policy_evade(oracle: ArenaOracle, idx: int = 1, noise=0.3):
    """Flee the nearest other mallet — the evader in pursuit mode."""
    rng = oracle.rng
    def act():
        b = oracle.mallets[idx].position
        others = [m for j, m in enumerate(oracle.mallets) if j != idx]
        if not others:
            return (rng.uniform(-1, 1), rng.uniform(-1, 1))
        c = min(others, key=lambda m: (m.position.x - b.x) ** 2 + (m.position.y - b.y) ** 2)
        dx, dy = b.x - c.position.x, b.y - c.position.y
        n = math.hypot(dx, dy) + 1e-6
        ax, ay = dx / n, dy / n
        ax += rng.normal(0, noise); ay += rng.normal(0, noise)
        return (ax, ay)
    return act


def policy_random(oracle: ArenaOracle, idx: int = 0):
    rng = oracle.rng
    def act():
        return (rng.uniform(-1, 1), rng.uniform(-1, 1))
    return act


# --------------------------------------------------------------------------
# Trajectory generation
# --------------------------------------------------------------------------

def rollout(cfg: ArenaConfig, seed: int, policies: str = "chase"):
    """Run one episode, return dict of frames (T,S,S,3) uint8, states (list),
    actions (T, n_mallets, 2), and metadata."""
    oracle = ArenaOracle(cfg, seed=seed)
    if policies == "random":
        pols = [policy_random(oracle, i) for i in range(cfg.n_mallets)]
    elif cfg.mode == "pursuit" and cfg.n_mallets >= 2:
        # mallet 0 chases, mallet 1 evades; extras chase
        pols = [policy_chase_puck(oracle, 0)] + [policy_evade(oracle, 1)]
        pols += [policy_chase_puck(oracle, i) for i in range(2, cfg.n_mallets)]
    elif cfg.mode == "airhockey" and cfg.n_mallets == 2:
        # striker (0) attacks the far goal; defender (1) guards it
        pols = [policy_strike_to_goal(oracle, 0), policy_defend_goal(oracle, 1, side="right")]
    else:
        # airhockey single (or >2): every mallet strikes the puck toward the goal
        pols = [policy_strike_to_goal(oracle, i) for i in range(cfg.n_mallets)]

    frames, states, actions = [], [], []
    frames.append(oracle.render())
    states.append(oracle.state())
    while not oracle.done:
        acts = [p() for p in pols]
        frame, state, done = oracle.step(acts)
        frames.append(frame)
        states.append(state)
        actions.append(acts)
    return {
        "frames": np.stack(frames).astype(np.uint8),
        "states": states,
        "actions": np.array(actions, dtype=np.float32),
        "field": list(oracle.field),
        "outcome": oracle.outcome,
        "seed": seed,
    }


def generate_dataset(cfg: ArenaConfig, n_episodes: int, out_dir: str,
                     policies: str = "chase", start_seed: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for e in range(n_episodes):
        seed = start_seed + e
        traj = rollout(cfg, seed=seed, policies=policies)
        np.savez_compressed(
            os.path.join(out_dir, f"ep_{e:05d}.npz"),
            frames=traj["frames"],
            actions=traj["actions"],
            field=np.array(traj["field"], dtype=np.float32),
        )
        # states as json (cheap, human-inspectable ground truth)
        with open(os.path.join(out_dir, f"ep_{e:05d}_states.json"), "w") as f:
            json.dump(traj["states"], f)
        manifest.append(dict(ep=e, seed=seed, T=len(traj["frames"]),
                             outcome=traj["outcome"], field=traj["field"]))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(dict(config=asdict(cfg), episodes=manifest), f, indent=2)
    return manifest


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _save_gif(frames, path, scale=4, fps=30):
    imgs = []
    for fr in frames:
        im = Image.fromarray(fr, "RGB")
        im = im.resize((fr.shape[1] * scale, fr.shape[0] * scale), Image.NEAREST)
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)


def main():
    ap = argparse.ArgumentParser(description="Air-hockey / pursuit oracle sim")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--mode", type=str, default="airhockey", choices=["airhockey", "pursuit"])
    ap.add_argument("--renderer", type=str, default="flat", choices=["flat", "iso"])
    ap.add_argument("--mallets", type=int, default=1)
    ap.add_argument("--hidden-field", action="store_true")
    ap.add_argument("--frame-size", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--policies", type=str, default="chase", choices=["chase", "random"])
    ap.add_argument("--preview-gif", type=str, default="")
    args = ap.parse_args()

    if args.mode == "pursuit" and args.mallets < 2:
        args.mallets = 2  # pursuit needs a chaser and an evader

    cfg = ArenaConfig(
        mode=args.mode, renderer=args.renderer, n_mallets=args.mallets,
        hidden_field=args.hidden_field, frame_size=args.frame_size,
        max_steps=args.max_steps,
    )

    if args.preview_gif:
        traj = rollout(cfg, seed=0, policies=args.policies)
        _save_gif(traj["frames"], args.preview_gif)
        print(f"wrote preview {args.preview_gif}  "
              f"({len(traj['frames'])} frames, outcome={traj['outcome']})")
        return

    manifest = generate_dataset(cfg, args.episodes, args.out, policies=args.policies)
    Ts = [m["T"] for m in manifest]
    outs = {}
    for m in manifest:
        outs[m["outcome"]] = outs.get(m["outcome"], 0) + 1
    print(f"generated {len(manifest)} episodes -> {args.out}")
    print(f"  frames/episode: min={min(Ts)} max={max(Ts)} mean={sum(Ts)/len(Ts):.0f}")
    print(f"  outcomes: {outs}")


if __name__ == "__main__":
    main()
