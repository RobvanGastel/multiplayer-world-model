"""Data batch types for the codec/world-model pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from wm.world_model.action_configs import ActionTensors


@dataclass
class VideoActionBatch:
    """One batch of video (+ optional per-step actions).

    video: (B, T, C, H, W), raw [0, 255] -- VideoCodec.preprocess_batch divides by 255 and
        resizes itself, so this is intentionally not pre-normalized.
    actions: per-step actions, or None -- wm.data.dataset's MatchClipDataset (the loader
        tools/train_codec.py uses) never populates this since CodecLoss doesn't read it; kept
        optional rather than required so video-only use isn't blocked on it. wm.data.dataset's
        create_loader (the world model's loader) does populate it, with an ActionTensors rather
        than a plain Tensor since LatentWorldModel reads it via ActionTensors.slice_time.
    """

    video: Tensor | None
    actions: ActionTensors | None = None
    # Pre-encoded raw codec latents (B, T_latent, C, h, w) in place of `video` -- see
    # tools/data/encode_latents.py and LatentWorldModel.latents_from_batch.
    latents: Tensor | None = None

    def to(self, device) -> "VideoActionBatch":
        return VideoActionBatch(
            video=self.video.to(device) if self.video is not None else None,
            actions=self.actions.to(device) if self.actions is not None else None,
            latents=self.latents.to(device) if self.latents is not None else None,
        )

    def slice_time(self, start: int, end: int, fps: int | None = None) -> "VideoActionBatch":
        """Slice `.video` to frames [start, end). `.actions` is left as-is -- it's windowed
        separately via its own ActionTensors.slice_time, since actions run at a possibly different
        step rate than video (see LatentWorldModel.forward/inference).

        fps is accepted for interface symmetry with ActionTensors.slice_time's frame-rate-aware
        windowing, but wm.data.dataset's loaders always store clips at the model's own target
        video.fps (no resampling), so it's unused here -- passing a mismatched fps will silently
        misbehave rather than resample.
        """
        return VideoActionBatch(video=self.video[:, start:end], actions=self.actions)
