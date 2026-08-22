"""Train the RAEv2 temporal-downsampling video codec.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from wm.codec.model import VideoCodec
from wm.codec.loss import CodecLoss
from wm.data.dataset import create_dataloader
from wm.utils import load_config

def train_codec(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    model: VideoCodec = VideoCodec(SimpleNamespace(encoder=args.encoder, decoder=args.decoder))
    model.train().to(device)

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
    iter_train_loader = iter(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # No lr_scheduler yet -- constant `args.lr` for now. The original sketch here called a
    # WarmupConstantCosineDecayLR that doesn't exist anywhere in this codebase (ported-over
    # reference, like CheckpointManager/training_tracker/DistributedEMA below); write a real one
    # (and re-add the compiled optimizer_step() that stepped it) when warmup/decay is needed.

    # with display_execution_time("Warming up dataloader", print_output=is_main_process):
    #     # First batch takes longer; warm it up before the timed loop.
    #     next(iter_train_loader)

    # checkpoint_manager = CheckpointManager(
    #     raw_model,
    #     checkpoint_dir=cfg.run.output_dir,
    #     save_every=cfg.run.checkpoint_every,
    #     keep_recent=cfg.run.checkpoint_keep_recent,
    #     keep_permanent_every=cfg.run.checkpoint_keep_permanent_every,
    #     total_steps=int(cfg.run.steps),
    #     model_ema_decay=cfg.optim.model_ema_decay,
    # )
    # ema_latent_mean = DistributedEMA(decay=cfg.run.latents_ema_decay, device=device)
    # ema_latent_std = DistributedEMA(decay=cfg.run.latents_ema_decay, initial_value=1.0, device=device)
    # checkpoint_manager.register(
    #     {
    #         "optimizer": optimizer,
    #         "lr_scheduler": lr_scheduler,
    #         "ema_latent_mean": ema_latent_mean,
    #         "ema_latent_std": ema_latent_std,
    #     }
    # )

    # start_step = _resume(cfg, checkpoint_manager, ema_latent_mean, ema_latent_std)

    # losses: dict[str, torch.Tensor] = {}
    # iter_num = start_step - 1  # so the final save below is well-defined even if the loop never runs
    # for iter_num in range(start_step, int(cfg.run.steps)):
    #     step_start_time = time.monotonic()

    #     batch = next(iter_train_loader).to(device)

    #     optimizer.zero_grad(set_to_none=True)
    #     with _autocast(device):
    #         model_outputs = model(batch)
    #         # The losses compute DINO embeddings, so keep autocast active here too.
    #         losses = loss(model_outputs, global_step=iter_num)

    #     with torch.no_grad():
    #         z = model_outputs.z.float()  # from bfloat16 to float32
    #         ema_latent_mean.update(z)
    #         ema_latent_std.update(z.std(keepdim=True))

    #     losses["loss_total"].backward()
    #     optimizer_step()
    #     checkpoint_manager.model_ema.step()

    #     training_tracker.on_batch_processed(batch, losses)

    #     early_logging_steps = 10
    #     if periodic_event(iter_num, cfg.run.log_every, cfg.run.steps) or iter_num < early_logging_steps:
    #         # All ranks must call get_stats()/compute() so the inner all_reduce completes.
    #         stats = training_tracker.get_stats(step=iter_num)
    #         stats["train/learning_rate"] = optimizer.param_groups[0]["lr"]
    #         stats["train/latent_mean"] = ema_latent_mean.compute()
    #         stats["train/latent_std"] = ema_latent_std.compute()
    #         if is_main_process:
    #             stats["System/step_ms"] = (time.monotonic() - step_start_time) * 1000
    #             stats |= {f"grad_norm/{k}": v.item() for k, v in loss.backward_metrics.items()}
    #             logger.info(f"Step {iter_num}: total loss {stats['train/loss_total']:.4f}")
    #             _wandb_log(stats, step=iter_num)

    #     if periodic_event(
    #         iter_num, cfg.validation.val_every, cfg.run.steps, include_0=cfg.validation.val_first
    #     ):
    #         with checkpoint_manager.model_ema.average_parameters():
    #             run_validation(cfg, device, raw_model, iter_val_loader, loss, iter_num)

    #     if periodic_event(iter_num, cfg.run.checkpoint_every, cfg.run.steps, include_0=False):
    #         if is_main_process:
    #             checkpoint_manager.maybe_save_checkpoint(
    #                 iter_num,
    #                 extra_data=_extra_data(iter_num, losses, ema_latent_mean, ema_latent_std),
    #             )
    #         if is_distributed:
    #             dist.barrier()

    #     if is_distributed:
    #         dist.barrier()

    # if is_main_process and iter_num >= start_step:  # skip when resuming an already-finished run
    #     checkpoint_manager.maybe_save_checkpoint(
    #         iter_num, extra_data=_extra_data(iter_num, losses, ema_latent_mean, ema_latent_std), final=True
    #     )
    # logger.info("Done training")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--encoder-config", type=str, default="configs/codec/rae_encoder.yml",
                     help="RAEEncoder config -- see wm.codec.rae_encoder.RAEEncoder")
    ap.add_argument("--loss-config", type=str, default="configs/codec/codec_loss.yml",
                     help="CodecLoss weights -- see wm.codec.loss.CodecLoss")
    ap.add_argument("--decoder-config", type=str, default="configs/codec/vit_decoder.yml",
                     help="ViTVideoDecoder config -- see wm.codec.vit_decoder.ViTVideoDecoder")
    ap.add_argument("--data-root", type=str, default="/PATH/REDACTED",
                     help="dataset root written by tools/collect_matches.py -- must contain "
                          "manifest_*.jsonl shards and a matches/ dir")
    ap.add_argument("--clip-len", type=int, default=16, help="frames per sampled video clip")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    args.encoder = load_config(args.encoder_config)
    args.loss = load_config(args.loss_config)
    args.decoder = load_config(args.decoder_config)

    train_codec(args)
