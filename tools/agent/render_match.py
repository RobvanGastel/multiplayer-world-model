import argparse
import functools
import random

from wm.env.arena import ArenaConfig
from wm.env.render_match import (
    ROUND_SECONDS, load_policy, policy_trained, play_match,
)
from wm.env.render import save_gif


def render_match(args):
    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
    print(f"seed={seed}")

    shot_clock_steps = int(ROUND_SECONDS / ArenaConfig().dt)  # 10s shot clock per round
    cfg = ArenaConfig(n_mallets=args.mallets, max_steps=shot_clock_steps,
                       frame_size=args.frame_size)

    striker = defender = None
    if args.policy:
        ppo = load_policy(args.policy, cfg)
        striker = functools.partial(policy_trained, ppo=ppo, deterministic=not args.stochastic)
        if args.policy2:
            ppo2 = load_policy(args.policy2, cfg)
            defender = functools.partial(
                policy_trained, ppo=ppo2, deterministic=not args.stochastic, mirror=True)
        elif args.self_play:
            defender = functools.partial(
                policy_trained, ppo=ppo, deterministic=not args.stochastic, mirror=True)


    frames, final_score = play_match(
        cfg, seed, args.px, args.fps, args.hold_secs,
        striker=striker, defender=defender
    )
    save_gif(frames, args.out, args.fps)
    print(f"wrote {args.out} ({len(frames)} frames, final score P0 {final_score[0]} - "
          f"{final_score[1]} P1)")


if __name__ == "__main__":
        ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                     help="omit for a fresh random match every run (the seed used is printed "
                          "so you can pass it back to reproduce that exact match)")
    ap.add_argument("--out", type=str, default="match.gif")
    ap.add_argument("--mallets", type=int, default=2)
    ap.add_argument("--frame-size", type=int, default=128,
                     help="native render resolution of a single view (actual detail)")
    ap.add_argument("--px", type=int, default=128,
                     help="displayed size of a single view (nearest-neighbor scaled up "
                          "from --frame-size; equal to it by default, i.e. no upscale)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold-secs", type=float, default=1.0, help="post-round score hold")
    ap.add_argument("--calibrate", action="store_true",
                     help="instead of a match, render one MIN-speed and one MAX-speed round")
    ap.add_argument("--policy", type=str, default="",
                     help="path to a PPO checkpoint (wm/policies/ppo.py) to control the "
                          "striker (mallet 0) instead of the scripted policy_strike_to_goal")
    ap.add_argument("--policy2", type=str, default="",
                     help="path to a second, different PPO checkpoint to control the defender "
                          "(mallet 1) -- for comparing two checkpoints head to head. Its view is "
                          "mirrored left/right same as --self-play; takes precedence over "
                          "--self-play if both are given")
    ap.add_argument("--stochastic", action="store_true",
                     help="with --policy/--policy2, sample actions instead of using each "
                          "policy's deterministic mean")
    ap.add_argument("--self-play", action="store_true",
                     help="with --policy, also control the defender (mallet 1) with the same "
                          "checkpoint instead of the scripted policy_defend_goal -- its view of "
                          "the board is mirrored left/right (and its action mirrored back) so it "
                          "genuinely attacks the opposite goal rather than racing mallet 0 for "
                          "the same one")
    args = ap.parse_args()

    render_match(args)
