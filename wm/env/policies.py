from __future__ import annotations
import math

from wm.env.arena import ArenaOracle

"""Scripted policies (to generate diverse, skillful data without a human)."""


def _predict_bounced_y(y0: float, vy: float, t: float, height: float) -> float:
    """Where a puck at y0 moving at vy in the y-direction will be after time t,
    accounting for elastic reflection off the top (y=0) and bottom (y=height)
    walls"""
    if height <= 0:
        return y0
    y = y0 + vy * t
    period = 2.0 * height
    y_mod = y % period
    if y_mod > height:
        y_mod = period - y_mod
    return y_mod


def policy_defend_goal(oracle: ArenaOracle, idx: int = 1, side: str = "right", noise=0.2,
                        advance: bool = False, aggression: float = 1.0,
                        defense_only: bool = False):
    """Guard a goal: hold a defensive line just in front of the goal, track the
    puck's y to stay between it and the goal, and clear the puck (drive into it,
    away from the goal) when it gets close. advance=False holds x at the fixed
    defensive line no matter what"""
    cfg = oracle.cfg
    rng = oracle.rng
    goal_x = cfg.width if side == "right" else 0.0
    # defensive line sits a bit in front of the goal
    line_x = cfg.width * 0.82 if side == "right" else cfg.width * 0.18
    clear_dist = (cfg.mallet_radius + cfg.puck_radius) * 3.0 * aggression
    goal_buffer = cfg.mallet_radius * 1.5  # how close to the mouth advance will chase

    approach_vx = 0.0625 * cfg.puck_max_speed

    def act():
        b = oracle.mallets[idx].position
        p = oracle.puck.position
        pv = oracle.puck.velocity
        dist = math.hypot(p.x - b.x, p.y - b.y)

        puck_threatening = (abs(p.x - goal_x) < cfg.width * 0.45)
        if not defense_only and dist < clear_dist and puck_threatening:
            # clear: hit the puck away from goal
            gx = goal_x - p.x
            cx, cy = -gx, (p.y - b.y)
            n = math.hypot(cx, cy) + 1e-6
            ax, ay = cx / n, cy / n
        else:
            # hold the line and cover the goal MOUTH. Predict where the puck will
            # cross the defensive x-line and get there first.
            approaching = (pv.x > approach_vx) if side == "right" else (pv.x < -approach_vx)
            if approaching and abs(pv.x) > 1e-3:
                # time for puck to reach the line, then its y at that time --
                # accounting for any top/bottom wall bounces between now and
                # then, not just a straight-line extrapolation
                tcross = (line_x - p.x) / pv.x
                tcross = max(0.0, min(tcross, 1.5))
                ty = _predict_bounced_y(p.y, pv.y, tcross, cfg.height)
            else:
                ty = p.y
            # clamp to the goal mouth so the defender always covers the opening
            ty = max(oracle.goal["y0"], min(oracle.goal["y1"], ty))
            if advance and dist < clear_dist:
                # chase the puck's own depth once it's past the line
                if side == "right":
                    tx = max(line_x, min(p.x, goal_x - goal_buffer))
                else:
                    tx = min(line_x, max(p.x, goal_x + goal_buffer))
            else:
                tx = line_x
            dx, dy = tx - b.x, ty - b.y
            n = math.hypot(dx, dy) + 1e-6
            ax, ay = dx / n, dy / n
        ax += rng.normal(0, noise); ay += rng.normal(0, noise)
        return (ax, ay)
    return act


DEFEND_VARIANTS: dict[str, dict] = {
    "standard":   dict(noise=0.20, advance=False, aggression=1.0, defense_only=False),
    "turtle":     dict(noise=0.05, advance=False, aggression=0.4, defense_only=True),
    "wall":       dict(noise=0.10, advance=True,  aggression=0.6, defense_only=True),
    "aggressive": dict(noise=0.15, advance=True,  aggression=1.8, defense_only=False),
    "erratic":    dict(noise=0.50, advance=True,  aggression=1.0, defense_only=False),
}


def make_defend_variant(oracle: ArenaOracle, idx: int = 1, side: str = "right",
                         variant: str = "standard"):
    """policy_defend_goal built from one of DEFEND_VARIANTS by name."""
    return policy_defend_goal(oracle, idx=idx, side=side, **DEFEND_VARIANTS[variant])


def policy_strike_to_goal(oracle: ArenaOracle, idx: int = 0, noise=0.25, home_side="left"):
    """Position behind the puck relative to the goal, then strike through it
    when it's actually reachable; otherwise fall back to guarding the goal
    """
    cfg = oracle.cfg
    rng = oracle.rng
    goal = (cfg.width - cfg.wall_thickness, cfg.height * 0.5)
    guard_home = policy_defend_goal(oracle, idx=idx, side=home_side, noise=noise, advance=True)
    # puck must be on our side of this line to be worth attacking, and not
    # already moving away.
    engage_x = cfg.width * 0.55
    departing_vx = 0.125 * cfg.puck_max_speed

    def act():
        b = oracle.mallets[idx].position
        p = oracle.puck.position
        pv = oracle.puck.velocity

        gettable = (p.x < engage_x) and not (pv.x > departing_vx)
        if not gettable:
            return guard_home()

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
            n = math.hypot(bx, by) + 1e-6
            ax, ay = bx / n + rng.normal(0, noise), by / n + rng.normal(0, noise)
        else:
            # in position and committing to the strike
            ax, ay = gux, guy
        return (ax, ay)
    return act


def policy_random(oracle: ArenaOracle, idx: int = 0):
    rng = oracle.rng
    def act():
        return (rng.uniform(-1, 1), rng.uniform(-1, 1))
    return act


def make_policies(oracle: ArenaOracle, policies: str = "chase"):
    """Pick a policy per mallet based on mallet-count. 'chase' picks the
    role-appropriate scripted policy for each mallet; 'random' ignores roles."""
    cfg = oracle.cfg
    if policies == "random":
        return [policy_random(oracle, i) for i in range(cfg.n_mallets)]
    if cfg.n_mallets == 2:
        # striker (0) attacks the far goal; defender (1) guards it
        return [policy_strike_to_goal(oracle, 0), policy_defend_goal(oracle, 1, side="right")]
    # single mallet (or >2): every mallet strikes the puck toward the goal
    return [policy_strike_to_goal(oracle, i) for i in range(cfg.n_mallets)]
