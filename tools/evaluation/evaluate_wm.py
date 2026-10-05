"""Diagnose a trained (multiplayer) world model on held-out clips, beyond the teacher-forced val loss.

Reports, per rollout horizon (latent steps past the context):
  * latent MSE vs the ground-truth latents, next to a "freeze the last context latent" baseline --
    a model that doesn't beat this baseline isn't modelling motion at all;
  * puck tracking in decoded frames (the puck is the only (240, 220, 90) yellow body, see
    wm/env/render_iso.py): presence rate and centroid error vs the decoded ground-truth latents,
    so codec error is factored out (the codec floor is reported separately);
  * action sensitivity: latent distance to a rollout with shuffled / zeroed actions, divided by the
    distance between two rollouts with the true actions but different noise seeds. ~1 means the
    model ignores its action input;
  * a small sampler sweep (n_diffusion_steps, noise_level).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from einops import rearrange

from wm.data.batch import VideoActionBatch
from wm.data.dataset import create_multiplayer_loader
from wm.world_model.multi_wrapper_world_model import MultiWrapperWorldModel

PUCK_RGB = (240, 220, 90)
# Mallet colors are fixed per mallet in every view (wm/env/render_iso.py's mcol): p0 drives the blue
# mallet, p1 the red one -- so each player's own mallet is findable in its own view.
OWN_MALLET_RGB = ((90, 160, 240), (240, 110, 110))


def puck_centroid(video: torch.Tensor, tol: float = 45.0, min_px: int = 3, color=PUCK_RGB):
    """video (..., T, C, H, W) in [0, 1] -> (centroid (..., T, 2) in px, present (..., T) bool) of
    the pixels within `tol` of `color` (the puck by default)."""
    rgb = torch.tensor(color, device=video.device, dtype=video.dtype).view(3, 1, 1) / 255
    mask = ((video - rgb).abs().amax(dim=-3) < tol / 255).float()  # (..., T, H, W)
    count = mask.sum(dim=(-2, -1))
    h, w = mask.shape[-2:]
    ys = torch.arange(h, device=video.device, dtype=video.dtype).view(h, 1)
    xs = torch.arange(w, device=video.device, dtype=video.dtype).view(1, w)
    cy = (mask * ys).sum(dim=(-2, -1)) / count.clamp(min=1)
    cx = (mask * xs).sum(dim=(-2, -1)) / count.clamp(min=1)
    return torch.stack([cx, cy], dim=-1), count >= min_px


def fresh(batch: VideoActionBatch) -> VideoActionBatch:
    # preprocess_batch rescales .video in place, so every rollout needs its own copy.
    return VideoActionBatch(video=batch.video.clone(), actions=batch.actions)


def with_keys(batch: VideoActionBatch, key_presses: torch.Tensor) -> VideoActionBatch:
    actions = batch.actions.slice_time(None, None)
    actions.key_presses = key_presses
    return VideoActionBatch(video=batch.video.clone(), actions=actions)


@torch.no_grad()
def rollout(model, batch, cfg, seed):
    torch.manual_seed(seed)
    return model.inference(fresh(batch), config=cfg, progress_bar=False).z_t  # (b p) t h w c


def per_step(x: torch.Tensor, n_ctx: int) -> list[float]:
    """(b, t, ...) squared-error-like tensor -> mean per latent step past the context."""
    return x[:, n_ctx:].flatten(2).mean(dim=(0, 2)).tolist()


@torch.no_grad()
def diagnose(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiWrapperWorldModel.load_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    swm = model.single_world_model
    n_ctx, td = swm.n_context_latents, swm.temporal_downsampling

    clip_len = model.config.n_context_frames + args.rollout_latents * td
    loader = create_multiplayer_loader(
        root_dir=args.data_root, clip_len=clip_len, batch_size=args.num_games,
        actions_config=model.config.actions, num_workers=args.num_workers, shuffle=True, seed=args.seed,
    )
    batch = next(iter(loader)).to(device)

    # Ground-truth latents / decoded GT latents / raw frames, all per player: (b p) ...
    pre = fresh(batch)
    swm.codec.preprocess_batch(pre)
    z_gt = swm.encode_video(pre)
    n_lat = z_gt.shape[1]
    raw = pre.video[:, : n_lat * td]
    gt_dec = swm.decode_to_video(z_gt)[:, : n_lat * td]

    def puck_stats(z_pred):
        dec = swm.decode_to_video(z_pred[:, :n_lat])
        c_pred, p_pred = puck_centroid(dec)
        c_gt, p_gt = puck_centroid(gt_dec)
        err = (c_pred - c_gt).norm(dim=-1)
        both = p_pred & p_gt
        # frames -> latent steps (mean over the td frames each latent decodes to)
        grp = lambda x: rearrange(x.float(), "b (t k) -> b t k", k=td).mean(-1)
        err_sum = grp(torch.where(both, err, torch.zeros_like(err)))
        both_n = grp(both)
        return {
            "puck_err_px": (err_sum[:, n_ctx:].sum(0) / both_n[:, n_ctx:].sum(0).clamp(min=1e-6)).tolist(),
            # of frames where the GT has a puck, how often the prediction also has one
            "puck_recall": (grp(both)[:, n_ctx:].sum(0) / grp(p_gt)[:, n_ctx:].sum(0).clamp(min=1e-6)).tolist(),
            # predicted puck where there is none in GT (hallucinated / duplicated)
            "puck_false_pos": (grp(p_pred & ~p_gt)[:, n_ctx:].mean(0)).tolist(),
        }

    def own_mallet_err(z_pred):
        """Per latent step: px error of each player's own mallet vs the decoded GT. On clips with
        random moves the mallet path isn't predictable from context, so this is low only when the
        rollout follows the given keys."""
        dec = swm.decode_to_video(z_pred[:, :n_lat])
        errs, valid = [], []
        for p, color in enumerate(OWN_MALLET_RGB):
            c_pred, p_pred = puck_centroid(dec[p::model.n_players], color=color)
            c_gt, p_gt = puck_centroid(gt_dec[p::model.n_players], color=color)
            both = p_pred & p_gt
            errs.append(torch.where(both, (c_pred - c_gt).norm(dim=-1), torch.zeros_like(c_gt[..., 0])))
            valid.append(both.float())
        grp = lambda x: rearrange(torch.cat(x), "b (t k) -> b t k", k=td).sum(-1)[:, n_ctx:].sum(0)
        return (grp(errs) / grp(valid).clamp(min=1e-6)).tolist()

    base_cfg = SimpleNamespace(n_diffusion_steps=10, noise_level=0.2, schedule_type="linear_quadratic")
    report: dict = {"checkpoint": args.checkpoint, "n_context_latents": n_ctx,
                    "temporal_downsampling": td, "rollout_latents": n_lat - n_ctx,
                    "num_games": args.num_games}

    # Codec floor: puck error of decoded GT latents vs raw frames.
    c_raw, p_raw = puck_centroid(raw)
    c_rec, p_rec = puck_centroid(gt_dec)
    both = p_raw & p_rec
    report["codec_floor"] = {
        "puck_err_px": ((c_raw - c_rec).norm(dim=-1)[both].mean().item() if both.any() else None),
        "puck_recall": (both.sum() / p_raw.sum().clamp(min=1)).item(),
        "raw_puck_visible": p_raw.float().mean().item(),
    }

    # 1) main rollout vs. freeze-last-frame baseline
    z_a = rollout(model, batch, base_cfg, seed=0)
    static = z_gt.clone()
    static[:, n_ctx:] = z_gt[:, n_ctx - 1: n_ctx]
    report["latent_mse"] = per_step((z_a - z_gt) ** 2, n_ctx)
    report["latent_mse_static_baseline"] = per_step((static - z_gt) ** 2, n_ctx)
    report["puck"] = puck_stats(z_a)
    report["puck_static_baseline"] = puck_stats(static)

    # 2) action sensitivity
    z_b = rollout(model, batch, base_cfg, seed=1)
    kp = batch.actions.key_presses
    kp_games = rearrange(kp, "(b p) t k -> b p t k", p=model.n_players)
    shuffled = rearrange(kp_games.roll(1, dims=0), "b p t k -> (b p) t k")
    z_shuf = rollout(model, with_keys(batch, shuffled), base_cfg, seed=0)
    z_zero = rollout(model, with_keys(batch, torch.zeros_like(kp)), base_cfg, seed=0)
    d_seed = per_step((z_a - z_b) ** 2, n_ctx)
    report["action_sensitivity"] = {
        "d_seed": d_seed,
        "ratio_shuffled": [s / max(d, 1e-8) for s, d in zip(per_step((z_a - z_shuf) ** 2, n_ctx), d_seed)],
        "ratio_zeroed": [s / max(d, 1e-8) for s, d in zip(per_step((z_a - z_zero) ** 2, n_ctx), d_seed)],
        "frac_keys_pressed": kp.float().mean().item(),
    }
    # Controllability: does each player's own mallet follow its keys? Compare against rollouts
    # driven by another game's / no keys and the freeze-frame baseline, all vs the same GT.
    report["own_mallet_err_px"] = {
        "true_actions": own_mallet_err(z_a), "shuffled_actions": own_mallet_err(z_shuf),
        "zeroed_actions": own_mallet_err(z_zero), "freeze_frame": own_mallet_err(static),
    }

    # 3) sampler sweep
    sweep = {}
    for steps in (10, 30):
        for noise in (0.2, 0.0, None):
            cfg = replace_ns(base_cfg, n_diffusion_steps=steps, noise_level=noise)
            z = z_a if (steps, noise) == (10, 0.2) else rollout(model, batch, cfg, seed=0)
            mse = per_step((z - z_gt) ** 2, n_ctx)
            ps = puck_stats(z)
            sweep[f"steps={steps},noise={noise}"] = {
                "latent_mse_first": mse[0], "latent_mse_last": mse[-1],
                "puck_err_mean": sum(ps["puck_err_px"]) / len(ps["puck_err_px"]),
                "puck_recall_mean": sum(ps["puck_recall"]) / len(ps["puck_recall"]),
            }
    report["sampler_sweep"] = sweep

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print_summary(report)
    print(f"wrote {out}")


def replace_ns(ns: SimpleNamespace, **kw) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(ns), **kw})


def print_summary(r: dict) -> None:
    fmt = lambda xs, k=4: " ".join(f"{x:.{k}f}" if isinstance(x, float) else str(x) for x in xs)
    idx = sorted({0, 1, 3, 7, 15, r["rollout_latents"] - 1} & set(range(r["rollout_latents"])))
    pick = lambda xs: [xs[i] for i in idx]
    print(f"\n=== {r['checkpoint']}  (context {r['n_context_latents']} latents, td={r['temporal_downsampling']})")
    print(f"codec floor: {r['codec_floor']}")
    print(f"horizon (latent step):     {fmt([i + 1 for i in idx])}")
    print(f"latent MSE  model:         {fmt(pick(r['latent_mse']))}")
    print(f"latent MSE  freeze-frame:  {fmt(pick(r['latent_mse_static_baseline']))}")
    print(f"puck err px model:         {fmt(pick(r['puck']['puck_err_px']), 2)}")
    print(f"puck err px freeze-frame:  {fmt(pick(r['puck_static_baseline']['puck_err_px']), 2)}")
    print(f"puck recall model:         {fmt(pick(r['puck']['puck_recall']), 2)}")
    print(f"puck false-pos model:      {fmt(pick(r['puck']['puck_false_pos']), 2)}")
    a = r["action_sensitivity"]
    print(f"action ratio shuffled:     {fmt(pick(a['ratio_shuffled']), 2)}   (~0 = actions ignored)")
    print(f"action ratio zeroed:       {fmt(pick(a['ratio_zeroed']), 2)}")
    if "own_mallet_err_px" in r:
        m = r["own_mallet_err_px"]
        print(f"own mallet err px, true actions:     {fmt(pick(m['true_actions']), 2)}")
        print(f"own mallet err px, shuffled actions: {fmt(pick(m['shuffled_actions']), 2)}   (>> true = follows keys)")
        print(f"own mallet err px, zeroed actions:   {fmt(pick(m['zeroed_actions']), 2)}")
        print(f"own mallet err px, freeze-frame:     {fmt(pick(m['freeze_frame']), 2)}")
    print("sampler sweep:")
    for k, v in r["sampler_sweep"].items():
        print(f"  {k:24s} " + "  ".join(f"{kk}={vv:.4f}" for kk, vv in v.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data-root", type=str)
    ap.add_argument("--rollout-latents", type=int, default=30, help="latent steps to roll out past the context")
    ap.add_argument("--num-games", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--out", type=str, default="runs/diagnose/report.json")
    diagnose(ap.parse_args())
