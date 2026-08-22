from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np
import pymunk


@dataclass
class ArenaConfig:
    scale: float = 3.0
    width: float = 200.0
    height: float = 80.0
    wall_thickness: float = 0.3

    # Object settings
    mallet_radius: float = 8.668
    mallet_mass: float = 3.0
    puck_radius: float = 6.5
    puck_mass: float = 1.0

    # dynamics settings
    puck_elasticity: float = 0.88
    wall_elasticity: float = 0.98
    mallet_elasticity: float = 0.85 # only governs mallet<->puck bounce
    mallet_bounce_elasticity: float = 0.0 # mallet<->wall and mallet<->mallet
    puck_friction: float = 0.02 # near-frictionless plane
    linear_damping: float = 0.15 # gentle global drag so energy bleeds off
    thrust: float = 1500.0 # force magnitude per action step
    mallet_max_speed: float = 366.667
    puck_max_speed: float = 533.334
    hidden_field: bool = False # constant force field on the puck
    hidden_field_max: float = 200.0 # sampled magnitude when enabled; 120*(5/3),
    dt: float = 1.0 / 60.0
    substeps: int = 2 # physics substeps per env step

    frame_size: int = 64
    n_mallets: int = 1 # 1 = single-agent; 2 = head-to-head
    max_steps: int = 400

    def __post_init__(self):
        # Lengths scale by s.
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

class ArenaOracle:
    """Deterministic air-hockey arena simulator."""

    def __init__(self, cfg: ArenaConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self._build()

    def _build(self):
        cfg = self.cfg
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 1.0 - cfg.linear_damping  # pymunk damping is a retain factor

        self._add_walls()
        self.mallets = [self._add_mallet(i) for i in range(cfg.n_mallets)]
        self.puck = self._add_puck()

        # override elasticity for mallet<->wall and mallet<->mallet contacts
        def _no_mallet_bounce(arbiter, space, data):
            arbiter.restitution = cfg.mallet_bounce_elasticity  # pymunk's Arbiter
        self.space.on_collision(2, 1, pre_solve=_no_mallet_bounce)
        self.space.on_collision(2, 2, pre_solve=_no_mallet_bounce)

        # hidden latent force field (sampled once per episode, applied to puck)
        if cfg.hidden_field:
            ang = self.rng.uniform(0, 2 * math.pi)
            mag = self.rng.uniform(0.3, 1.0) * cfg.hidden_field_max
            self.field = (math.cos(ang) * mag, math.sin(ang) * mag)
        else:
            self.field = (0.0, 0.0)

        # goal region for airhockey: the goal IS the wall line.
        gh = cfg.height * 0.5
        self.goal = dict(
            x0=cfg.width, x1=cfg.width,   # the right wall line
            y0=(cfg.height - gh) / 2, y1=(cfg.height + gh) / 2,
        )
        self.steps = 0
        self.done = False
        self.outcome = None  # 'goal' | 'conceded' | 'timeout'

        # measured acceleration (delta-v / dt over the last step()), zero
        # until the first step -- see step()'s prev_vel bookkeeping.
        self.accel_puck = (0.0, 0.0)
        self.accel_mallets = [(0.0, 0.0) for _ in self.mallets]

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
        if cfg.n_mallets == 2:
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

        ang = self.rng.uniform(0, 2 * math.pi)
        speed = self.rng.uniform(0.02083, 0.0625) * cfg.puck_max_speed
        body.velocity = (math.cos(ang) * speed, math.sin(ang) * speed)
        shape = pymunk.Circle(body, cfg.puck_radius)
        shape.elasticity = cfg.puck_elasticity
        shape.friction = cfg.puck_friction
        shape.collision_type = 3
        self.space.add(body, shape)
        return body

    def _clamp_speed(self, body, max_speed):
        v = body.velocity
        s = math.hypot(v.x, v.y)
        if s > max_speed:
            body.velocity = (v.x / s * max_speed, v.y / s * max_speed)

    def step(self, actions):
        """actions: list of (ax, ay) in [-1,1] per mallet. Returns (state, done)."""
        cfg = self.cfg
        if len(actions) != len(self.mallets):
            raise ValueError(f"expected {len(self.mallets)} actions, got {len(actions)}")

        prev_vel_puck = self.puck.velocity
        prev_vel_mallets = [m.velocity for m in self.mallets]

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

        self.accel_puck = ((self.puck.velocity.x - prev_vel_puck.x) / cfg.dt,
                            (self.puck.velocity.y - prev_vel_puck.y) / cfg.dt)
        self.accel_mallets = [((m.velocity.x - pv.x) / cfg.dt, (m.velocity.y - pv.y) / cfg.dt)
                               for m, pv in zip(self.mallets, prev_vel_mallets)]

        self.steps += 1
        self._check_terminal()
        return self.state(), self.done

    def _confine_mallets(self):
        """Keep mallets inside the playfield and, in 2-mallet airhockey, each in
        its own half. Mallets never leave through the goal gaps - those are for
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
        if cfg.n_mallets != 2:
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

    def _check_terminal(self):
        cfg = self.cfg
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
            # opponent scored (puck reached this side's goal; not necessarily a literal own-goal touch)
            self.outcome = "conceded"
        if self.steps >= cfg.max_steps and not self.done:
            self.done = True
            self.outcome = "timeout"

    def state(self) -> dict:
        """Ground-truth low-dim internal state - puck/mallet pose+velocity, the
        hidden field, and step count. This is what a world model should learn
        to predict/represent; it is not itself fed to a pixel-only model.

        "*_accel" is measured delta-v/dt over the last step() call (zero at
        episode start) -- kept as separate keys rather than appended onto the
        existing [x,y,vx,vy] lists so callers unpacking those stay unaffected."""
        return {
            "puck": [self.puck.position.x, self.puck.position.y,
                     self.puck.velocity.x, self.puck.velocity.y],
            "puck_accel": list(self.accel_puck),
            "mallets": [[b.position.x, b.position.y, b.velocity.x, b.velocity.y]
                        for b in self.mallets],
            "mallet_accels": [list(a) for a in self.accel_mallets],
            "field": list(self.field),
            "step": self.steps,
        }

    @staticmethod
    def flat_transform(cfg: ArenaConfig, S: int | None = None, margin_frac: float = 0.0):
        """(sc, ox, oy) mapping board coords -> flat-view pixels: px = o + wc*sc.
        Shared by render_flat() and anything (tests, overlays) that needs to
        place a marker at an exact board position in that same pixel space -
        a pure function of cfg, so it doesn't need a live oracle instance."""
        S = cfg.frame_size if S is None else S
        eff_width = cfg.width * (1.0 + 2.0 * margin_frac)
        sc = min(S / eff_width, S / cfg.height)
        ox = (S - cfg.width * sc) / 2
        oy = (S - cfg.height * sc) / 2
        return sc, ox, oy

    def render_flat(self, margin_frac: float = 0.0) -> np.ndarray:
        """Top-down (S,S,3) uint8 frame. Cheapest view; also what render_iso.py's
        camera derives its own frame from, so both views stay consistent."""
        cfg = self.cfg
        S = cfg.frame_size
        img = np.zeros((S, S, 3), dtype=np.uint8)
        img[:] = (12, 14, 20)
        # uniform scale (letterbox) so circles + positions share one scale ->
        # contact renders true; center the board in the square frame
        sc, ox, oy = self.flat_transform(cfg, S, margin_frac)

        def X(wx): return ox + wx * sc
        def Y(wy): return oy + wy * sc

        # playfield panel
        x0, y0b = int(X(0)), int(Y(0))
        x1, y1b = int(X(cfg.width)), int(Y(cfg.height))
        img[y0b:y1b, x0:x1] = (26, 30, 42)

        g = self.goal
        gy0, gy1 = int(Y(g["y0"])), int(Y(g["y1"]))

        # goal pockets: small panels extending behind each goal line
        if margin_frac > 0:
            pocket_depth = cfg.width * margin_frac
            px0 = max(0, int(X(-pocket_depth)))
            px1 = min(S, int(X(cfg.width + pocket_depth)))
            img[gy0:gy1, px0:x0] = (22, 26, 35)   # left pocket (blue mallet's goal)
            img[gy0:gy1, x1:px1] = (22, 26, 35)   # right pocket (red mallet's goal)

        # center line exactly at board middle
        midx = int(X(cfg.width * 0.5))
        img[y0b:y1b, midx:midx + 1] = (60, 66, 86)
        # both goals -- fixed by mallet color, not board side, so a goal
        # stays the same color across every view (see render_iso.py's
        # _goal_colors): red mallet (mallets[1]) always gets the orange goal,
        # blue mallet (mallets[0]) always gets the green one.
        img[gy0:gy1, x1 - 2:x1] = (150, 120, 70)     # right, red mallet's goal = orange
        img[gy0:gy1, x0:x0 + 2] = (70, 150, 100)     # left, blue mallet's goal = green

        # goal creases -- painted floor markings only, no physics meaning.
        g_mid = (g["y0"] + g["y1"]) / 2.0
        g_r = (g["y1"] - g["y0"]) / 2.0
        # matches each goal's own color (left=green/blue mallet, right=orange/red mallet)
        self._arc(img, 0.0, g_mid, g_r, -math.pi / 2, math.pi / 2, X, Y, (70, 150, 100))
        self._arc(img, cfg.width, g_mid, g_r, math.pi / 2, 3 * math.pi / 2, X, Y, (150, 120, 70))

        p = self.puck.position
        self._disc(img, X(p.x), Y(p.y), cfg.puck_radius * sc, (240, 220, 90))
        mcolors = [(90, 160, 240), (240, 110, 110)]
        for i, b in enumerate(self.mallets):
            self._disc(img, X(b.position.x), Y(b.position.y),
                       cfg.mallet_radius * sc, mcolors[i % len(mcolors)])
        return img

    @staticmethod
    def _arc(img, cx_w, cy_w, r_w, theta0, theta1, X, Y, color, n=64, thickness=2):
        """Thin arc of a world-space circle (center cx_w,cy_w, radius r_w),
        sampled as points and mapped through the caller's board->pixel
        transform (X, Y)"""
        S = img.shape[0]
        for i in range(n + 1):
            theta = theta0 + (theta1 - theta0) * i / n
            px = X(cx_w + r_w * math.cos(theta))
            py = Y(cy_w + r_w * math.sin(theta))
            xi, yi = int(px), int(py)
            for dy in range(-(thickness // 2), thickness - thickness // 2):
                for dx in range(-(thickness // 2), thickness - thickness // 2):
                    yy, xx = yi + dy, xi + dx
                    if 0 <= yy < S and 0 <= xx < S:
                        img[yy, xx] = color

    @staticmethod
    def _disc(img, cx, cy, r, color):
        S = img.shape[0]
        r = max(1.0, r)
        x0, x1 = max(0, int(cx - r)), min(S, int(cx + r + 1))
        y0, y1 = max(0, int(cy - r)), min(S, int(cy + r + 1))
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                if (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r:
                    img[yy, xx] = color
