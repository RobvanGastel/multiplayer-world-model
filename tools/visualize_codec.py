"""Reconstruct real clips through a trained codec checkpoint and save
input-vs-reconstruction stills/GIFs, to see what the latents actually encode.

Usage:
    python -m tools.visualize_codec --checkpoint runs/codec.pt --out-dir runs/codec_preview
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wm.codec.model import VideoCodec
from wm.data.dataset import create_dataloader


def denormalize(video: torch.Tensor) -> np.ndarray:
    """VideoCodec.normalize_video's inverse: [-1, 1] float -> [0, 255] uint8, (T, H, W, C)."""
    video = ((video * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
    return video.permute(0, 2, 3, 1).cpu().numpy()


def side_by_side(input_frames: np.ndarray, output_frames: np.ndarray) -> list[Image.Image]:
    gap = 4
    h, w = input_frames.shape[1:3]
    frames = []
    for inp, out in zip(input_frames, output_frames):
        combo = Image.new("RGB", (w * 2 + gap, h), (10, 11, 15))
        combo.paste(Image.fromarray(inp, "RGB"), (0, 0))
        combo.paste(Image.fromarray(out, "RGB"), (w + gap, 0))
        frames.append(combo)
    return frames


def visualize(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model = VideoCodec.load_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    loader = create_dataloader(args.data_root, clip_len=args.clip_len, batch_size=args.num_clips,
                                num_workers=0, shuffle=True)
    batch = next(iter(loader)).to(device)

    with torch.no_grad():
        out = model(batch)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(batch.video.shape[0]):
        input_frames = denormalize(out.input_video[i])
        output_frames = denormalize(out.output_video[i])
        frames = side_by_side(input_frames, output_frames)

        gif_path = out_dir / f"clip{i}.gif"
        upscaled = [f.resize((f.width * args.upscale, f.height * args.upscale), Image.NEAREST)
                    for f in frames]
        upscaled[0].save(gif_path, save_all=True, append_images=upscaled[1:], duration=100, loop=0)

        mid_path = out_dir / f"clip{i}_mid.png"
        frames[len(frames) // 2].resize(
            (frames[0].width * args.upscale, frames[0].height * args.upscale), Image.NEAREST
        ).save(mid_path)
        print(f"clip {i}: wrote {gif_path} and {mid_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data-root", type=str, default="")
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--num-clips", type=int, default=4, help="how many clips to reconstruct")
    ap.add_argument("--upscale", type=int, default=3, help="nearest-neighbor upscale for visibility")
    ap.add_argument("--out-dir", type=str, default="runs/codec_preview")
    ap.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    visualize(ap.parse_args())
