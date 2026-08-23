"""Data batch types for the codec/world-model pipeline."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class VideoActionBatch:
    """One batch of video (+ optional per-step actions).

    video: (B, T, C, H, W), raw [0, 255] -- VideoCodec.preprocess_batch divides by 255 and
        resizes itself, so this is intentionally not pre-normalized.
    actions: (B, T, ...) per-step actions, or None -- wm.data.dataset's MatchClipDataset (the
        loader tools/train_codec.py uses) never populates this since CodecLoss doesn't read it;
        kept optional rather than required so video-only use isn't blocked on it.
    """

    video: Tensor
    actions: Tensor | None = None

    def to(self, device) -> "VideoActionBatch":
        return VideoActionBatch(
            video=self.video.to(device),
            actions=self.actions.to(device) if self.actions is not None else None,
        )
