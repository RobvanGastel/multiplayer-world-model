import argparse
import itertools
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import yaml

from wm.data.dataset import create_latent_multiplayer_loader, create_loader, create_multiplayer_loader
from wm.utils import load_config, merge_config
from wm.training.lr_schedule import WarmupConstantCosineDecayLR
from wm.world_model.latent_world_model import LatentWorldModel
from wm.world_model.multi_wrapper_world_model import MultiWrapperWorldModel


def cycle_loader(loader):
    """Yield batches forever, restarting a new epoch each time the dataset is exhausted -- same
    helper as tools/train_codec.py's."""
    while True:
        yield from loader


def _autocast(device: torch.device):
    """bfloat16 autocast on CUDA, a no-op elsewhere (so the trainer runs on CPU too).
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def run_validation(model: LatentWorldModel, val_loader, device: torch.device, n_batches: int,
                    iter_num: int) -> dict[str, float]:
    """Average the world-model forward loss over up to n_batches of the held-out val set.

    Fresh `iter(val_loader)` each call, capped with itertools.islice rather than next() so a val
    set smaller than n_batches * batch_size degrades gracefully instead of raising StopIteration.
    Returns the per-key means (e.g. "loss_total") so the caller can track a best checkpoint.
    """
    model.eval()
    totals: dict[str, float] = {}
    n_seen = 0
    with torch.no_grad():
        for batch in itertools.islice(val_loader, n_batches):
            batch = batch.to(device)
            with _autocast(device):
                losses = model(batch)
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n_seen += 1
    model.train()

    means = {k: v / n_seen for k, v in totals.items()}
    print(f"step {iter_num + 1}: validation  "
          + "  ".join(f"val_{k}={v:.4f}" for k, v in means.items()), flush=True)
    return means


def _write_model_config_yaml(architecture_config_path: str, save_path: str) -> None:
    with open(architecture_config_path) as f:
        architecture_dict = yaml.safe_load(f)

    config_path = Path(save_path).parent / LatentWorldModel.CONFIG_FILENAME
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.safe_dump({"model": {"architecture": {"config": architecture_dict}}}, f)


def train_worldmodel(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if hasattr(args.architecture, "n_players"):
        model = MultiWrapperWorldModel(args.architecture)
    else:
        model = LatentWorldModel(args.architecture)
    model.train().to(device)

    _write_model_config_yaml(args.architecture_config, args.save_path)

    train_loader, val_loader, metrics_loader = _create_dataloaders(args, args.metrics, model)
    iter_train_loader = cycle_loader(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                   betas=(args.adam_beta1, args.adam_beta2))
    lr_scheduler = WarmupConstantCosineDecayLR(
        optimizer, warmup_steps=args.warmup_steps, constant_steps=args.constant_steps,
        decay_steps=args.decay_steps, min_lr=args.min_lr,
    )

    # save_path is always overwritten in place (the "latest" checkpoint); best_path is only
    # overwritten when val loss improves -- same {root}_best{ext} convention as ppo.py's
    # best-model tracking, so runs/ only ever holds these two files instead of one per save_every.
    best_path = Path(args.save_path).with_stem(Path(args.save_path).stem + "_best")
    best_val_loss = float("inf")
    start_iter = 0

    # schedule from step 0 -- for continuing an older checkpoint saved without optimizer state
    # (use a short warmup: the fresh AdamW moments make full-LR first steps unstable).
    resume, init_from = getattr(args, "resume", None), getattr(args, "init_from", None)
    if resume and Path(resume).exists():
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_iter = checkpoint["iter_num"] + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"resumed {resume} at step {start_iter}, lr={lr_scheduler.get_last_lr()[0]:.2e}", flush=True)
    elif init_from:
        checkpoint = torch.load(init_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"initialized weights from {init_from} (its step {checkpoint.get('iter_num')})", flush=True)

    def training_state(iter_num: int) -> dict:
        return {"iter_num": iter_num, "best_val_loss": best_val_loss,
                "optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict()}

    losses: dict[str, torch.Tensor] = {}
    iter_num = start_iter - 1
    for iter_num in range(start_iter, args.steps):
        step_start_time = time.monotonic()

        batch = next(iter_train_loader).to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            losses = model(batch)

        losses["loss_total"].backward()
        optimizer.step()
        lr_scheduler.step()

        if (iter_num + 1) % args.log_every == 0:
            print(f"step {iter_num + 1}/{args.steps}  loss_total={losses['loss_total'].item():.4f}  "
                  f"lr={lr_scheduler.get_last_lr()[0]:.2e}  "
                  f"step_ms={(time.monotonic() - step_start_time) * 1000:.0f}", flush=True)

        if args.val_every and (iter_num + 1) % args.val_every == 0:
            val_means = run_validation(model, val_loader, device, args.validation_n_batches, iter_num)
            if val_means["loss_total"] < best_val_loss:
                best_val_loss = val_means["loss_total"]
                model.save_checkpoint(best_path, extra_data={"iter_num": iter_num, "val_loss_total": best_val_loss})
                print(f"step {iter_num + 1}: new best val_loss_total={best_val_loss:.4f} -> saved {best_path}",
                      flush=True)

        if args.save_every and (iter_num + 1) % args.save_every == 0:
            model.save_checkpoint(args.save_path, extra_data=training_state(iter_num))

    model.save_checkpoint(args.save_path, extra_data=training_state(iter_num))
    print("done training", flush=True)


def _create_dataloaders(args, metrics, model: LatentWorldModel | MultiWrapperWorldModel):
    """Build the (train, val, metrics) dataloaders. The val/metrics loaders use fixed seeds so the
    same held-out subsample is scored every eval. train_root/test_root are separate collected-data
    directories (see configs/world_model/train.yml) rather than a split within one directory --
    generate held-out data with tools/agent/collect_matches.py --out-dir <test_root>."""
    # Apply the eval's context override before deriving the metrics clip length, so the loader, the
    # rollout, and the metric indexing all agree on n_context_frames (see set_inference_context).
    if metrics.n_context_frames is not None:
        model.set_inference_context(metrics.n_context_frames)

    common = dict(actions_config=model.config.actions, num_workers=args.num_workers)
    stride = metrics.eval_temporal_downsampling or model.temporal_downsampling
    # MultiWrapperWorldModel needs paired p0/p1 batches (see wm/data/dataset.py's
    # MultiPlayerActionClipDataset); its `n_players` attribute is the trainer's signal to switch.
    loader_fn = create_multiplayer_loader if hasattr(model, "n_players") else create_loader

    # latents: true trains on <root>/latents from tools/data/encode_latents.py -- no codec pass and
    # no video I/O per step. The metrics loader stays on video, as rollouts are compared in pixels.
    if getattr(args, "latents", False):
        train_fn = partial(create_latent_multiplayer_loader, n_latents=model.config.video.timesteps // model.temporal_downsampling)
    else:
        train_fn = partial(loader_fn, clip_len=model.config.video.timesteps)

    train_loader = train_fn(
        root_dir=args.train_root,
        batch_size=args.batch_size,
        seed=args.seed,
        **common,
    )
    # Pre-encoded latents are raw codec outputs, only meaningful for the codec that made them.
    codec_used = getattr(train_loader.dataset, "codec_checkpoint", model.config.codec_checkpoint)
    assert codec_used == model.config.codec_checkpoint, (
        f"latents under {args.train_root} were encoded with {codec_used}, model uses {model.config.codec_checkpoint}")
    val_loader = train_fn(
        root_dir=args.test_root,
        batch_size=args.validation_batch_size or args.batch_size,
        seed=37,
        **common,
    )
    metrics_loader = loader_fn(
        # video loaders take a single dir -- use the first when test_root lists several
        root_dir=args.test_root[0] if isinstance(args.test_root, list) else args.test_root,
        clip_len=model.config.n_context_frames + metrics.num_unrolled_frames * stride,
        batch_size=metrics.per_device_batch_size,
        seed=38,
        **common,
    )
    return train_loader, val_loader, metrics_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/world_model/train.yml",
                         help="yaml of training-loop defaults -- see configs/world_model/train.yml")
    parser.add_argument("--architecture-config", type=str, default="configs/world_model/latent_world_model.yml",
                         help="LatentWorldModelConfig -- see wm.world_model.latent_world_model.LatentWorldModel")
    parser.add_argument("--metrics-config", type=str, default="configs/world_model/metrics.yml",
                         help="eval-rollout config -- see configs/world_model/metrics.yml")
    parser.add_argument("--resume", type=str, default=None,
                         help="checkpoint saved by this trainer to continue exactly (skipped if missing)")
    parser.add_argument("--init-from", type=str, default=None,
                         help="checkpoint to take weights from; schedule/optimizer start fresh")
    parser.add_argument("--save-path", dest="save_path", type=str, default=None,
                         help="override the config's save_path")
    raw_args = parser.parse_args()
    args = merge_config(raw_args.config, raw_args)

    if args.decay_steps is None:
        args.decay_steps = max(0, args.steps - args.warmup_steps - args.constant_steps)

    args.architecture = load_config(args.architecture_config)
    args.metrics = load_config(args.metrics_config)

    train_worldmodel(args)