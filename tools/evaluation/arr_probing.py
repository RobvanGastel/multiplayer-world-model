"""Action Recoverability Ratio (ARR), MIRA's controllability metric (Section 6.2 / Appendix D of
arXiv 2607.05352), adapted to this repo's 2-player W/A/S/D air hockey.

An action probe -- a frozen DINOv3 encoder with a small attention-pooling head -- is trained on real
video to detect which keys a player presses within an 8-frame window of its own view (per-key binary
cross-entropy against the logged key presses). For a world-model rollout conditioned on the real
actions, the probe is slid over the generated frames and over the codec reconstruction of the same
clip, and each key's detections are scored with average precision (AP) against the commanded keys:

    ARR(key) = AP_gen(key) / AP_recon(key),   ARR = mean over keys, bootstrap CI over games.

ARR = 1: actions are as recoverable from the generation as from a faithful reconstruction; below 1
the model under-renders them. Dividing by the reconstruction cancels the probe's own imperfection
and the codec's visual domain.

Usage:
    python -m tools.evaluation.arr_probing train-probe --out runs/arr_probe/probe.pt
    python -m tools.evaluation.arr_probing eval --probe runs/arr_probe/probe.pt --checkpoint runs/wm/world_model.pt
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import ConcatDataset, DataLoader

from wm.codec.dino import DinoModel
from wm.data.batch import VideoActionBatch
from wm.data.dataset import ActionClipDataset, collate_action, create_multiplayer_loader
from wm.world_model.multi_wrapper_world_model import MultiWrapperWorldModel

KEYS = ("w", "a", "s", "d")
WINDOW = 8  # frames per probe window, as in MIRA (0.4 s there at 20 fps; 0.13 s here at 60 fps)


def window_labels(key_presses: torch.Tensor) -> torch.Tensor:
    """(..., WINDOW - 1, 4) actions driving a window's transitions -> (..., 4) "pressed within the
    window" targets (action t drives frame t -> t + 1)."""
    return key_presses.amax(dim=-2).float()


class ActionProbe(nn.Module):
    """Frozen DINOv3 + attention pooling: one learned query per key attends over all patch tokens of
    the window's frames (with learned temporal and spatial position embeddings, since the direction
    of motion is what identifies a key), then a linear readout per key."""

    def __init__(self, dino_model: str, weights_dir: str, grid: tuple[int, int], n_keys: int = 4,
                 n_heads: int = 8):
        super().__init__()
        self.dino = DinoModel(dino_model, last_layer_only=True, compile=False, weights_dir=weights_dir)
        dim = self.dino.dino_dim
        self.time_embed = nn.Parameter(0.02 * torch.randn(WINDOW, 1, dim))
        self.space_embed = nn.Parameter(0.02 * torch.randn(1, grid[0] * grid[1], dim))
        self.queries = nn.Parameter(0.02 * torch.randn(n_keys, dim))
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.readout = nn.Linear(dim, 1)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("dino.")]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (B, WINDOW, 3, H, W) in [0, 1] -> key logits (B, 4)."""
        with torch.no_grad():
            feats = self.dino.dino_forward(frames)[-1]  # (B, T, C, h, w)
        tokens = rearrange(feats, "b t c h w -> b t (h w) c") + self.time_embed + self.space_embed
        tokens = self.norm(rearrange(tokens, "b t n c -> b (t n) c"))
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(queries, tokens, tokens)
        return self.readout(pooled).squeeze(-1)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, as sklearn's average_precision_score)."""
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / labels.sum())


def per_key_ap(scores: np.ndarray, labels: np.ndarray) -> list[float]:
    return [average_precision(scores[:, k], labels[:, k]) for k in range(len(KEYS))]


def load_probe(path: str, device) -> ActionProbe:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    probe = ActionProbe(ckpt["dino_model"], ckpt["weights_dir"], tuple(ckpt["grid"]))
    probe.load_state_dict(ckpt["state_dict"])
    return probe.to(device).eval()


def window_loader(roots: list[str], batch_size: int, seed: int, num_workers: int, actions_config) -> DataLoader:
    """Random 8-frame windows from every player-view clip of `roots` (real frames + logged keys)."""
    dataset = ConcatDataset([ActionClipDataset(root, WINDOW) for root in roots])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                      generator=torch.Generator().manual_seed(seed), pin_memory=True,
                      collate_fn=partial(collate_action, actions_config=actions_config))


def train_probe(args) -> None:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    actions_config = argparse.Namespace(valid_keys=list(KEYS))
    train = window_loader(args.train_roots, args.batch_size, args.seed, args.num_workers, actions_config)
    first = next(iter(train))
    h, w = first.video.shape[-2:]
    grid = (h // 16, w // 16)
    probe = ActionProbe(args.dino_model, args.weights_dir, grid).to(device)
    opt = torch.optim.AdamW(probe.head_parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    batches = itertools.chain([first], itertools.chain.from_iterable(itertools.repeat(train)))
    for step, batch in zip(range(args.steps), batches):
        frames = batch.video.to(device, non_blocking=True).float() / 255
        labels = window_labels(batch.actions.key_presses.to(device))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(probe(frames).float(), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print(f"step {step + 1}/{args.steps}  bce={loss.item():.4f}", flush=True)

    # Calibration on real held-out video (MIRA: 0.84 mAP over its nine controls).
    probe.eval()
    test = window_loader(args.test_roots, args.batch_size, args.seed + 1, args.num_workers, actions_config)
    scores, labels = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in itertools.islice(test, args.test_batches):
            scores.append(probe(batch.video.to(device).float() / 255).float().cpu())
            labels.append(window_labels(batch.actions.key_presses))
    aps = per_key_ap(torch.cat(scores).numpy(), torch.cat(labels).numpy())
    print("held-out real video AP per key: " + "  ".join(f"{k.upper()}={a:.3f}" for k, a in zip(KEYS, aps))
          + f"  mAP={np.nanmean(aps):.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": probe.state_dict(), "dino_model": args.dino_model, "weights_dir": args.weights_dir,
                "grid": grid, "heldout_ap": aps, "train_roots": args.train_roots}, out)
    print(f"wrote {out}")


def probe_windows(probe: ActionProbe, video: torch.Tensor, starts: list[int]) -> torch.Tensor:
    """video (N, T, 3, H, W) in [0, 1] -> probe probabilities (N, len(starts), 4)."""
    windows = torch.stack([video[:, s:s + WINDOW] for s in starts], dim=1)  # (N, S, W, 3, H, W)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = probe(rearrange(windows, "n s t c h w -> (n s) t c h w").float())
    return rearrange(torch.sigmoid(logits.float()), "(n s) k -> n s k", n=video.shape[0])


@torch.no_grad()
def evaluate(args) -> None:
    device = torch.device("cuda")
    probe = load_probe(args.probe, device)
    model = MultiWrapperWorldModel.load_from_checkpoint(args.checkpoint, device=device).eval()
    swm = model.single_world_model
    n_ctx_frames = swm.n_context_latents * swm.temporal_downsampling
    clip_len = model.config.n_context_frames + args.rollout_latents * swm.temporal_downsampling
    loader = create_multiplayer_loader(args.data_root, clip_len, args.games_per_batch, model.config.actions,
                                       num_workers=args.num_workers, shuffle=True, seed=args.seed)
    cfg = argparse.Namespace(n_diffusion_steps=10, noise_level=0.2, schedule_type="linear_quadratic")

    per_game = {"gen": [], "recon": [], "labels": [], "recon_labels": []}  # (games, players * windows, 4)
    for batch_idx, batch in enumerate(itertools.islice(loader, args.num_batches)):
        batch = batch.to(device)
        real_keys = batch.actions.key_presses
        commanded = batch.actions
        if args.actions == "swapped":
            # Hybrid clips (MIRA's human action-adherence protocol): each game's context with the next
            # game's key sequence, so predicting "what this agent would do" no longer matches the
            # commanded keys -- only following them does.
            commanded = batch.actions.slice_time(None, None)
            games = rearrange(real_keys, "(b p) t k -> b p t k", p=model.n_players)
            commanded.key_presses = rearrange(games.roll(1, dims=0), "b p t k -> (b p) t k")
        torch.manual_seed(args.seed + batch_idx)
        outputs = model.inference(VideoActionBatch(video=batch.video.clone(), actions=commanded),
                                  config=cfg, progress_bar=False)
        gen = swm.decode_to_video(outputs.z_t)  # (b p) t c h w in [0, 1]
        recon = swm.decode_to_video(swm.encode_video(outputs.preprocessed_batch))
        n_frames = min(gen.shape[1], recon.shape[1])
        # windows over the generated part only (after the real context)
        starts = list(range(n_ctx_frames, n_frames - WINDOW + 1, args.window_stride))
        to_labels = lambda keys: rearrange(
            torch.stack([window_labels(keys[:, s:s + WINDOW - 1]) for s in starts], dim=1).cpu(),
            "(b p) s k -> b (p s) k", p=model.n_players)
        for name, video in (("gen", gen), ("recon", recon)):
            probs = probe_windows(probe, video[:, :n_frames], starts).cpu()
            per_game[name].append(rearrange(probs, "(b p) s k -> b (p s) k", p=model.n_players))
        # generation is scored against the keys it was given; the reconstruction ceiling against
        # the real keys of the real clip it reconstructs
        per_game["labels"].append(to_labels(commanded.key_presses))
        per_game["recon_labels"].append(to_labels(real_keys))
        print(f"batch {batch_idx + 1}/{args.num_batches}", flush=True)

    gen, recon, labels, recon_labels = (torch.cat(per_game[k]).numpy()
                                        for k in ("gen", "recon", "labels", "recon_labels"))

    def arr_of(idx):
        flat = lambda x: x[idx].reshape(-1, len(KEYS))
        ap_gen, ap_recon = per_key_ap(flat(gen), flat(labels)), per_key_ap(flat(recon), flat(recon_labels))
        return ap_gen, ap_recon, [g / r for g, r in zip(ap_gen, ap_recon)]

    n_games = gen.shape[0]
    ap_gen, ap_recon, arr = arr_of(np.arange(n_games))
    rng = np.random.default_rng(args.seed)
    boot = [np.nanmean(arr_of(rng.integers(0, n_games, n_games))[2]) for _ in range(args.bootstrap)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    report = {"checkpoint": args.checkpoint, "data_root": args.data_root, "actions": args.actions, "games": n_games,
              "windows_per_game": gen.shape[1], "keys": KEYS, "ap_gen": ap_gen, "ap_recon": ap_recon,
              "arr_per_key": arr, "arr": float(np.nanmean(arr)), "arr_ci95": [float(lo), float(hi)],
              "key_frequency": labels.reshape(-1, len(KEYS)).mean(0).tolist()}
    print(f"\n=== {args.checkpoint} on {Path(args.data_root).name}, {args.actions} actions: {n_games} games")
    for k, g, r, a, f in zip(KEYS, ap_gen, ap_recon, arr, report["key_frequency"]):
        print(f"  {k.upper()}  AP_gen={g:.3f}  AP_recon={r:.3f}  ARR={a:.3f}  (pressed in {f:.0%} of windows = chance AP)")
    print(f"  ARR = {report['arr']:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tp = sub.add_parser("train-probe")
    tp.add_argument("--train-roots", nargs="+")
    tp.add_argument("--test-roots", nargs="+")
    tp.add_argument("--dino-model", default="dinov3_vitb16")
    tp.add_argument("--weights-dir")
    tp.add_argument("--steps", type=int, default=4000)
    tp.add_argument("--batch-size", type=int, default=64)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--test-batches", type=int, default=60)
    tp.add_argument("--num-workers", type=int, default=6)
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--out", default="runs/arr_probe/probe.pt")
    ev = sub.add_parser("eval")
    ev.add_argument("--probe", default="runs/arr_probe/probe.pt")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--data-root")
    ev.add_argument("--actions", choices=("real", "swapped"), default="real",
                    help="condition on the clip's real keys (MIRA's ARR) or on another game's keys "
                         "(hybrid clips: separates following the keys from predicting the agent)")
    ev.add_argument("--rollout-latents", type=int, default=30)
    ev.add_argument("--games-per-batch", type=int, default=16)
    ev.add_argument("--num-batches", type=int, default=6)
    ev.add_argument("--window-stride", type=int, default=4)
    ev.add_argument("--bootstrap", type=int, default=1000)
    ev.add_argument("--num-workers", type=int, default=4)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--out", default="")
    args = ap.parse_args()
    train_probe(args) if args.cmd == "train-probe" else evaluate(args)
