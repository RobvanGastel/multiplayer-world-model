"""Pre-encode collected matches into codec latents once, so world-model training never reads video
or runs the frozen codec (see wm/data/dataset.py's LatentMultiPlayerDataset, and `latents: true` in
the world-model train config).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wm.data.batch import VideoActionBatch
from wm.utils import load_config
from wm.world_model.latent_world_model import LatentWorldModel


@torch.no_grad()
def encode_frames(model: LatentWorldModel, frames: torch.Tensor, chunk: int) -> torch.Tensor:
    """(B, T, C, H, W) uint8 -> raw codec latents (B, T // td, C, h, w), encoded in time chunks.
    Chunks are multiples of td, so they line up with the codec's non-overlapping temporal conv."""
    td = model.temporal_downsampling
    frames = frames[:, : frames.shape[1] // td * td]
    out = []
    for start in range(0, frames.shape[1], chunk):
        batch = VideoActionBatch(video=frames[:, start:start + chunk].to(model.device))
        model.codec.preprocess_batch(batch)
        _, encoder_output = model.codec.encode(batch.video, trim_video=False)
        out.append(encoder_output.z.float())
    return torch.cat(out, dim=1)


def check_window_equivalence(model: LatentWorldModel, frames: torch.Tensor, chunk: int) -> float:
    """Max abs difference between a full-sequence encoding sliced at an even offset and encoding
    that window alone -- should be ~0 if slicing pre-encoded latents is exact."""
    td = model.temporal_downsampling
    window = model.config.video.timesteps
    start = 5 * td
    full = encode_frames(model, frames, chunk)[:, start // td:(start + window) // td]
    alone = encode_frames(model, frames[:, start:start + window], chunk)
    return (full - alone).abs().max().item() / alone.abs().max().item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = load_config(args.architecture_config)
    model = LatentWorldModel(architecture.wm_config).to(device).eval()
    assert args.chunk % model.temporal_downsampling == 0

    root = Path(args.root)
    out_dir = root / "latents"
    out_dir.mkdir(exist_ok=True)
    manifests = sorted(root.glob("manifest_*.jsonl"))[args.shard_index::args.num_shards]

    checked = False
    for manifest in manifests:
        out_path = out_dir / f"{manifest.stem}.pt"
        if out_path.exists():
            print(f"skip {out_path} (exists)", flush=True)
            continue
        games = []
        with open(manifest) as f:
            for line in f:
                for game in json.loads(line)["games"]:
                    payloads = [torch.load(root / "matches" / game[p]["file"], map_location="cpu",
                                           weights_only=False, mmap=True) for p in ("p0", "p1")]
                    frames = torch.stack([pl["frames"] for pl in payloads])
                    if not checked:
                        rel_err = check_window_equivalence(model, frames, args.chunk)
                        print(f"window-equivalence check: relative max abs diff {rel_err:.2e}", flush=True)
                        assert rel_err < 1e-3, "full-game encoding doesn't match per-window encoding"
                        checked = True
                    z = encode_frames(model, frames, args.chunk).half().cpu()
                    games.append({
                        "p0": game["p0"]["file"], "p1": game["p1"]["file"],
                        "z_p0": z[0].clone(), "z_p1": z[1].clone(),
                        "actions_p0": payloads[0]["actions_wasd"].to(torch.uint8).clone(),
                        "actions_p1": payloads[1]["actions_wasd"].to(torch.uint8).clone(),
                    })
        tmp = out_path.with_suffix(".tmp")
        torch.save({"codec_checkpoint": architecture.wm_config.codec_checkpoint,
                    "temporal_downsampling": model.temporal_downsampling, "games": games}, tmp)
        tmp.rename(out_path)
        print(f"wrote {out_path} ({len(games)} games)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="collected-data dir (manifest_*.jsonl + matches/)")
    ap.add_argument("--architecture-config", type=str, default="configs/world_model/latent_world_model.yml")
    ap.add_argument("--chunk", type=int, default=128, help="frames per encode call (multiple of td)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    main(ap.parse_args())
