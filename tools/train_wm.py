import argparse
import itertools
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from wm.data.dataset import create_loader
from wm.utils import load_config, merge_config
from wm.training.lr_schedule import WarmupConstantCosineDecayLR
from wm.world_model.latent_world_model import LatentWorldModel


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

    model: LatentWorldModel = LatentWorldModel(args.architecture)
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

    losses: dict[str, torch.Tensor] = {}
    for iter_num in range(args.steps):
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
            model.save_checkpoint(args.save_path, extra_data={"iter_num": iter_num})

    model.save_checkpoint(args.save_path, extra_data={"iter_num": iter_num})
    print("done training", flush=True)


def _create_dataloaders(args, metrics, model: LatentWorldModel):
    """Build the (train, val, metrics) dataloaders. The val/metrics loaders use fixed seeds so the
    same held-out subsample is scored every eval. train_root/test_root are separate collected-data
    directories (see configs/world_model/train.yml) rather than a split within one directory --
    generate held-out data with tools/data/collect_matches.py --out-dir <test_root>."""
    # Apply the eval's context override before deriving the metrics clip length, so the loader, the
    # rollout, and the metric indexing all agree on n_context_frames (see set_inference_context).
    if metrics.n_context_frames is not None:
        model.set_inference_context(metrics.n_context_frames)

    common = dict(actions_config=model.config.actions, num_workers=args.num_workers)
    stride = metrics.eval_temporal_downsampling or model.temporal_downsampling

    train_loader = create_loader(
        root_dir=args.train_root,
        clip_len=model.config.video.timesteps,
        batch_size=args.batch_size,
        seed=args.seed,
        **common,
    )
    val_loader = create_loader(
        root_dir=args.test_root,
        clip_len=model.config.video.timesteps,
        batch_size=args.validation_batch_size or args.batch_size,
        seed=37,
        **common,
    )
    metrics_loader = create_loader(
        root_dir=args.test_root,
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
    raw_args = parser.parse_args()
    args = merge_config(raw_args.config, raw_args)

    if args.decay_steps is None:
        args.decay_steps = max(0, args.steps - args.warmup_steps - args.constant_steps)

    args.architecture = load_config(args.architecture_config)
    args.metrics = load_config(args.metrics_config)

    train_worldmodel(args)