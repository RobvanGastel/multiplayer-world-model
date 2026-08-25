"""Train the RAEv2 temporal-downsampling video codec.
"""
from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from tools.visualize_codec import denormalize, side_by_side
from wm.codec.model import VideoCodec
from wm.codec.loss import CodecLoss
from wm.data.dataset import create_dataloader
from wm.training.ema import DistributedEMA
from wm.training.lr_schedule import WarmupConstantCosineDecayLR
from wm.utils import load_config, merge_config


def cycle_loader(loader):
    """Yield batches forever, restarting a new epoch each time the dataset is
    exhausted -- plain `iter(loader)` raises StopIteration once its one pass
    over the (small, ~340-match) dataset ends, which crashed training around
    step 800 the first time this ran."""
    while True:
        yield from loader


def visualize_step(model: VideoCodec, vis_loader, device: torch.device, out_dir: str,
                    iter_num: int, upscale: int) -> None:
    """Reconstruct a random sample and save input-vs-reconstruction GIFs, via
    tools/visualize_codec.py's own denormalize/side_by_side -- same output
    shape as running that script by hand, just on the live in-training model
    instead of a saved checkpoint. Restores model.train() before returning."""
    model.eval()
    batch = next(iter(vis_loader)).to(device)
    with torch.no_grad():
        out = model(batch)
    model.train()

    step_dir = Path(out_dir) / f"step{iter_num + 1:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for i in range(batch.video.shape[0]):
        frames = side_by_side(denormalize(out.input_video[i]), denormalize(out.output_video[i]))
        upscaled = [f.resize((f.width * upscale, f.height * upscale), Image.NEAREST) for f in frames]
        upscaled[0].save(step_dir / f"clip{i}.gif", save_all=True,
                          append_images=upscaled[1:], duration=100, loop=0)
    print(f"step {iter_num + 1}: wrote {batch.video.shape[0]} visualization clip(s) to {step_dir}",
          flush=True)


def _autocast(device: torch.device):
    """bfloat16 autocast on CUDA, a no-op elsewhere (so the trainer runs on CPU too).
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def train_codec(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    model: VideoCodec = VideoCodec(SimpleNamespace(encoder=args.encoder, decoder=args.decoder))
    model.train().to(device)

    resumed_latent_mean_std = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        resumed_latent_mean_std = checkpoint.get("latent_mean_std")
        print(f"resumed weights from {args.resume}", flush=True)

    loss = CodecLoss(args.loss)
    loss.to(device)

    if loss.weights.auto_weight:
        loss.bind_last_layer(model.decoder.last_layer_weight)
    if loss.weights.loss_dino_latent_consistency > 0:
        loss.bind_encoder_dino(model.encoder.rae_dino)

    if args.compile:
        model.compile()

    train_loader = create_dataloader(
        args.data_root,
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    iter_train_loader = cycle_loader(train_loader)

    # Separate from train_loader (own shuffled dataset instance) so pulling a
    # visualization batch doesn't consume/skew the training data stream.
    vis_loader = None
    if args.vis_every:
        vis_loader = create_dataloader(args.data_root, clip_len=args.clip_len,
                                        batch_size=args.vis_num_clips, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                   betas=(args.adam_beta1, args.adam_beta2))
    lr_scheduler = WarmupConstantCosineDecayLR(
        optimizer, warmup_steps=args.warmup_steps, constant_steps=args.constant_steps,
        decay_steps=args.decay_steps, min_lr=args.min_lr,
    )

    # First batch warm-up
    next(iter_train_loader)

    # Running mean/std of the encoder's latent output z, resumed from the checkpoint's saved
    # values if available, so a resumed run doesn't re-warm this from scratch.
    resumed_mean, resumed_std = resumed_latent_mean_std or (0.0, 1.0)
    ema_latent_mean = DistributedEMA(decay=args.latents_ema_decay, initial_value=resumed_mean, device=device)
    ema_latent_std = DistributedEMA(decay=args.latents_ema_decay, initial_value=resumed_std, device=device)

    losses: dict[str, torch.Tensor] = {}
    for iter_num in range(args.steps):
        step_start_time = time.monotonic()

        batch = next(iter_train_loader).to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            model_outputs = model(batch)
            # The losses compute DINO embeddings, so keep autocast active here too.
            losses = loss(model_outputs, global_step=iter_num)

        with torch.no_grad():
            z = model_outputs.z.float()  # from bfloat16 to float32
            ema_latent_mean.update(z)
            ema_latent_std.update(z.std())

        losses["loss_total"].backward()
        optimizer.step()
        lr_scheduler.step()

        if (iter_num + 1) % args.log_every == 0:
            print(f"step {iter_num + 1}/{args.steps}  loss_total={losses['loss_total'].item():.4f}  "
                  f"lr={lr_scheduler.get_last_lr()[0]:.2e}  "
                  f"step_ms={(time.monotonic() - step_start_time) * 1000:.0f}", flush=True)

        if args.save_every and (iter_num + 1) % args.save_every == 0:
            # requires a codec_config.yaml already sitting in a parent dir of
            # args.save_path -- see VideoCodec.save_checkpoint/_find_codec_config.
            model.save_checkpoint(args.save_path, extra_data={
                "latent_mean_std": (ema_latent_mean.compute(), ema_latent_std.compute()),
            })

        if args.vis_every and (iter_num + 1) % args.vis_every == 0:
            visualize_step(model, vis_loader, device, args.vis_dir, iter_num, args.vis_upscale)

    model.save_checkpoint(args.save_path, extra_data={
        "latent_mean_std": (ema_latent_mean.compute(), ema_latent_std.compute()),
    })
    print("done training")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/codec/train.yml",
                     help="yaml of training-loop defaults")
    ap.add_argument("--encoder-config", type=str, default="configs/codec/rae_encoder.yml",
                     help="RAEEncoder config -- see wm.codec.rae_encoder.RAEEncoder")
    ap.add_argument("--loss-config", type=str, default="configs/codec/codec_loss.yml",
                     help="CodecLoss weights -- see wm.codec.loss.CodecLoss")
    ap.add_argument("--decoder-config", type=str, default="configs/codec/vit_decoder.yml",
                     help="ViTVideoDecoder config -- see wm.codec.vit_decoder.ViTVideoDecoder")
    ap.add_argument("--resume", type=str, default=None,
                     help="path to a codec checkpoint (e.g. runs/codec.pt) to resume model weights")
    raw_args = ap.parse_args()
    args = merge_config(raw_args.config, raw_args)

    if args.decay_steps is None:
        args.decay_steps = max(0, args.steps - args.warmup_steps - args.constant_steps)

    args.encoder = load_config(args.encoder_config)
    args.loss = load_config(args.loss_config)
    args.decoder = load_config(args.decoder_config)

    train_codec(args)
