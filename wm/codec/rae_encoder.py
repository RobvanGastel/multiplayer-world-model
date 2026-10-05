"""RAEv2 encoder: a frozen DINOv3 backbone, layer aggregation, and a strided-conv bottleneck."""

from __future__ import annotations
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from wm.codec.dino import DINO_DIM, DinoModel
from wm.training.weights import init_weights


@dataclass
class RAEEncoderOutputs:
    z: Tensor
    dino_features: tuple[Tensor, ...] | None = None


class RAEEncoder(nn.Module):
    """Frozen DINOv3 backbone -> aggregation -> bottleneck -> latent."""

    def __init__(self, config, require_dino_weights: bool = True) -> None:
        """
        config: a plain object with this module's fields as attributes (latent_dim, rae_model,
            video, aggregation_layers, bottleneck.{stride,temporal_stride},
            compile_dino) -- see configs/codec/rae_encoder.yml and wm.utils.load_config, which
            builds one of these straight from that yaml (no pydantic model anymore).
        require_dino_weights: Can be set to False at inference when we'll load from a pretrained
            checkpoint. That way the deployment environment doesn't need a separate path to valid
            DINO weights.
        """
        super().__init__()
        self.config = config
        dino_dim = DINO_DIM[config.rae_model]

        # Strided 3D convolution: compresses temporal_stride frames of stride x stride DINOv3 patches
        # into one latent cell (kernel == stride, so windows don't overlap).
        stride = (config.bottleneck.temporal_stride, config.bottleneck.stride, config.bottleneck.stride)
        self.rae_projection = nn.Conv3d(dino_dim, config.latent_dim, kernel_size=stride, stride=stride, bias=True)

        # Initialise the bottleneck projection before building the frozen DINO backbone, so the
        # backbone's pretrained weights are not overwritten by `init_weights`.
        self.apply(init_weights)

        self.rae_dino = DinoModel(
            config.rae_model,
            last_layer_only=False,
            layer_indices=tuple(config.aggregation_layers),
            compile=config.compile_dino,
            require_pretrained=require_dino_weights,
            weights_dir=getattr(config, "dino_weights_dir", None),
        )

    def get_downsampling_factors(self) -> tuple[int, int]:
        return self.config.bottleneck.temporal_stride, 16 * self.config.bottleneck.stride

    def forward(self, video: Tensor) -> RAEEncoderOutputs:
        # VideoCodec normalizes to [-1, 1]; DinoModel expects [0, 1].
        video = (video + 1) / 2

        with torch.no_grad():
            features = self.rae_dino.dino_forward(video)  # list of (B, T, dino_dim, H, W)

        # Average the selected DINOv3 layers and add the last layer, then compress to the latent.
        agg = torch.stack(features, dim=0).mean(dim=0) + features[-1]
        z = self.rae_projection(rearrange(agg, "b t c h w -> b c t h w"))
        z = rearrange(z, "b c t h w -> b t c h w")

        return RAEEncoderOutputs(z=z, dino_features=tuple(features))
