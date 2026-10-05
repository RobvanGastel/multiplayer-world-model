"""Multiplayer wrapper around :class:`LatentWorldModel`, ported from MIRA's
``src/mira/world_model/multi_wrapper_world_model.py`` (github.com/mira-wm/mira) -- this repo is a
condensed port of that codebase, and MIRA already implements two-player (n_players) training the
same way this file does, so the tiling/action-combination scheme below is copied from there
rather than invented here.

:class:`MultiWrapperWorldModel` tiles ``n_players`` per-player clips into a single frame (stacked
vertically along the height) and processes them jointly with one inner :class:`LatentWorldModel`.
The frozen codec keeps running per player at the original (single-player) resolution, while the
diffusion transformer sees the full tiled grid. Because the transformer uses RoPE positional
encodings (no resolution-dependent learned parameters, see wm/world_model/layers/rope.py), a
single-player checkpoint can be warm-started into the tiled model even though the resolutions
differ -- see :meth:`load_state_dict`.

The players arrive contiguously and player-id-ordered as ``n_players`` rows of the batch (the data
loader's grouping invariant -- see wm/data/dataset.py's MultiPlayerActionClipDataset/
collate_multiplayer_action), so ``rearrange("(b p) ... -> b p ...")`` lines the rows up with the
tiling. The wrapper exposes the same surface as :class:`LatentWorldModel` (config, codec,
world_model, inference, visualize, ...) so it is a drop-in replacement in the trainer and metrics.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from wm.data.batch import VideoActionBatch
from wm.utils import to_namespace
from wm.world_model.action_configs import ActionTensors
from wm.world_model.latent_world_model import InferenceOutputs, LatentWorldModel

logger = logging.getLogger(__name__)


class MultiWrapperWorldModel(nn.Module):
    """Wraps a :class:`LatentWorldModel` so that ``n_players`` clips are tiled into a single frame.

    The players are stacked vertically (along the height); the codec runs for each player
    independently at the original (single-player) resolution.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.n_players = config.n_players

        # The inner world model's video resolution is configured for the tiled (multi-player) setup,
        # but the codec runs per player so it must keep the original (single-player) resolution. The
        # inner ``LatentWorldModel.__init__`` overwrites the codec resolution from its config, so we
        # multiply the height first, build the inner model, then reset the codec height afterwards.
        # Work on a deep copy so the caller's config object is not mutated (a reuse would
        # double-multiply).
        original_height = config.wm_config.video.height
        wm_config = copy.deepcopy(config.wm_config)
        wm_config.video.height *= config.n_players
        self.single_world_model = LatentWorldModel(wm_config)
        self.single_world_model.codec.config.encoder.video.height = original_height

        action_dim = self.single_world_model.action_encoder.dim
        self.player_embedding = nn.Parameter(torch.randn(config.n_players, action_dim) * 0.02)
        self.player_action_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(config.n_players * action_dim, action_dim)
        )

    @property
    def config(self):
        return self.single_world_model.config

    @property
    def codec(self):
        return self.single_world_model.codec

    @property
    def world_model(self):
        return self.single_world_model.world_model

    @property
    def device(self):
        return next(self.parameters()).device

    # Codec-derived factors the trainer/metrics read off the model directly; proxy them to the inner
    # world model so the wrapper stays a drop-in replacement.
    @property
    def temporal_downsampling(self) -> int:
        return self.single_world_model.temporal_downsampling

    @property
    def spatial_downsampling(self) -> int:
        return self.single_world_model.spatial_downsampling

    @property
    def n_context_latents(self) -> int:
        return self.single_world_model.n_context_latents

    def set_inference_context(self, n_context_frames: int) -> None:
        """Proxy the inference-context override to the inner world model (see LatentWorldModel)."""
        self.single_world_model.set_inference_context(n_context_frames)

    def unnormalize_tokens(self, z: Tensor) -> Tensor:
        return self.single_world_model.unnormalize_tokens(z)

    def decode_to_video(self, z: Tensor) -> Tensor:
        """Decode per-player latents ``(b*p, t, h, w, c)`` to per-player video; pass straight through."""
        return self.single_world_model.decode_to_video(z)

    def _combine_player_actions(self, a_flat: Tensor) -> Tensor:
        """Combine per-player encoded actions into one conditioning stream.

        Args:
            a_flat: ``(b*p, t_a, d)`` per-player encoded actions, players contiguous within each group.

        Returns:
            ``(b, t_a, d)`` combined actions: the player embedding is added to each player's actions,
            the players are then concatenated channel-wise (``p*d``) and projected back down to ``d``.
        """
        a = rearrange(a_flat, "(b p) t d -> b p t d", p=self.n_players)
        a = a + self.player_embedding[None, :, None, :]
        a = rearrange(a, "b p t d -> b t (p d)")
        return self.player_action_projection(a)

    def forward(self, batch: VideoActionBatch, *args, **kwargs) -> dict[str, Tensor]:
        swm = self.single_world_model
        z_flat = swm.latents_from_batch(batch, swm.config.video.timesteps)

        # (b*p, t, h, w, c) per-player latents -> tiled frame (b, t, p*h, w, c), stacked along height.
        z = rearrange(z_flat, "(b p) t h w c -> b t (p h) w c", p=self.n_players)

        # Slice actions with the SAME offset the single model uses (off = atd - 1) and the same
        # //td-aware n_action_steps, so training alignment matches the single-player path at td > 1.
        off = swm.action_temporal_downsampling - 1
        a_flat = swm.action_encoder(batch.actions.slice_time(off, swm.n_action_steps + off))
        a = self._combine_player_actions(a_flat)

        # Reuse the single-player training tail on the tiled z + combined actions; swm.bos is built at
        # the tiled resolution, so its z.shape[0] repeat matches z's layout directly.
        return swm.diffusion_loss(z, a)

    @torch.no_grad()
    def inference(
        self,
        batch: VideoActionBatch,
        config=None,
        progress_bar: bool = True,
    ) -> InferenceOutputs:
        from types import SimpleNamespace  # noqa: PLC0415

        if config is None:
            config = SimpleNamespace(n_diffusion_steps=10, noise_level=0.2, schedule_type="linear_quadratic")
        swm = self.single_world_model
        swm.codec.preprocess_batch(batch)
        batch = batch.to(self.device)

        z_flat = swm.encode_video(batch).clone()  # clone because of torch.compile
        z = rearrange(z_flat, "(b p) t h w c -> b t (p h) w c", p=self.n_players)

        z_t = torch.randn_like(z)
        z_t[:, : swm.n_context_latents] = z[:, : swm.n_context_latents]

        window_size = swm.n_context_latents + 1
        start = 0  # in case the loop does not run
        streaming_kv_caches = None

        for start in tqdm(
            range(0, z.shape[1] - window_size + 1),
            desc="Diffusion inference",
            disable=not progress_bar,
        ):
            # Action slice for this window, offset by atd-1 so actions align with the latent frames
            # exactly as in training (see forward) and in single-player inference.
            off = swm.action_temporal_downsampling - 1
            action_start = start * swm.action_temporal_downsampling + off
            action_end = (start + window_size - 1) * swm.action_temporal_downsampling + off
            a_flat_step = swm.action_encoder(
                batch.actions.slice_time(action_start, action_end),
            ).clone()  # clone because of torch.compile
            current_a = self._combine_player_actions(a_flat_step)
            z_t[:, start : start + window_size], streaming_kv_caches = swm.denoise_streaming(
                z_t[:, start : start + window_size],
                current_a,
                n_diffusion_steps=config.n_diffusion_steps,
                noise_level=config.noise_level,
                streaming_kv_caches=streaming_kv_caches,
                schedule_type=config.schedule_type,
            )

        n_generated_latents = start + window_size

        # Decode each player's view separately, then re-tile.
        z_t_split = rearrange(z_t, "b t (p h) w c -> (b p) t h w c", p=self.n_players)
        output_video_per_player = self.decode_to_video(z_t_split[:, :n_generated_latents])
        output_video = rearrange(output_video_per_player, "(b p) t c h w -> b t c (p h) w", p=self.n_players)

        return InferenceOutputs(
            output_video=output_video,
            preprocessed_batch=batch,
            z_t=z_t_split,
        )

    def init_streaming_inference(self, batch: VideoActionBatch) -> Tensor:
        """Encode the bootstrap batch into tiled latents for streaming inference.

        Args:
            batch: Per-player batch (first dim ``b*p``).

        Returns:
            Tiled latents ``(b, t, p*h, w, c)`` matching the wrapper's internal latent layout.
        """
        swm = self.single_world_model
        batch = batch.to(self.device)
        swm.codec.preprocess_batch(batch)
        z_flat = swm.encode_video(batch).clone()  # clone because of torch.compile
        return rearrange(z_flat, "(b p) t h w c -> b t (p h) w c", p=self.n_players)

    def streaming_inference_step(
        self,
        z: Tensor,
        actions_history: ActionTensors,
        streaming_kv_cache=None,
        config=None,
    ):
        """Multi-player counterpart of :meth:`LatentWorldModel`'s streaming inference step.

        ``actions_history`` has ``batch_size = b * n_players`` so each player's action stream is
        independent; they are combined via :meth:`_combine_player_actions` before the world-model
        forward pass.
        """
        from types import SimpleNamespace  # noqa: PLC0415

        if config is None:
            config = SimpleNamespace(n_diffusion_steps=10, noise_level=0.2, schedule_type="linear_quadratic")

        swm = self.single_world_model
        # Latent-space streaming: actions align via action_temporal_downsampling, so this is
        # temporal-downsampling-agnostic. The wrapper's tiling is spatial (height) and orthogonal.
        # Put noise in the last frame -- that's what we want to predict, the rest is context.
        z_t = torch.cat([z[:, 1:], torch.randn_like(z[:, :1])], dim=1)

        # Action slice offset by atd-1 so actions align with the latent frames as in training and
        # single-player streaming; the last atd-1 actions belong to the not-yet-predicted frame.
        off = swm.action_temporal_downsampling - 1
        n_action_steps = (z.shape[1] - 1) * swm.action_temporal_downsampling
        a_flat = swm.action_encoder(
            actions_history.slice_time(-n_action_steps - off, -off if off else None).to(self.device)
        )
        current_a = self._combine_player_actions(a_flat)

        return swm.denoise_streaming(
            z_t,
            current_a,
            streaming_kv_caches=streaming_kv_cache,
            n_diffusion_steps=config.n_diffusion_steps,
            noise_level=config.noise_level,
            schedule_type=config.schedule_type,
        )

    # Params that cannot be warm-started from a single-player checkpoint (new params, or params whose
    # shape depends on the tiled resolution); see :meth:`load_state_dict`.
    _WARMSTART_EXEMPT: tuple[str, ...] = (
        "player_embedding",
        "player_action_projection.",
        "single_world_model.bos",
        "single_world_model.action_encoder.mouse_temporal_pool.",
        "single_world_model.action_encoder.keyboard_temporal_pool.",
        # Per-player key-dropout token: a single-player checkpoint may not have it -> random init.
        "single_world_model.action_encoder.key_dropout_embed",
    )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load weights, warm-starting from a single-player checkpoint when needed.

        A single-player checkpoint (no ``single_world_model.`` prefix) is remapped under that prefix
        so every player starts from the same weights. Params in ``_WARMSTART_EXEMPT`` are allowed to
        keep their random init when they are absent from the checkpoint or have a mismatching shape.
        """
        if not any(k.startswith("single_world_model.") for k in state_dict):
            state_dict = {f"single_world_model.{k}": v for k, v in state_dict.items()}

        state_dict = dict(state_dict)
        reinitialised = []
        for k, v in self.state_dict().items():
            if k.startswith(self._WARMSTART_EXEMPT) and (
                k not in state_dict or state_dict[k].shape != v.shape
            ):
                state_dict[k] = v  # keep our init; makes the load a no-op for this param
                reinitialised.append(k)
        if reinitialised:
            logger.info(
                "Warm-start: keeping random init for %d expected param(s): %s",
                len(reinitialised),
                sorted(reinitialised),
            )

        # A single-player checkpoint may carry params this model intentionally lacks (e.g. the
        # whole-keyboard keyboard_dropout_token, replaced by key_dropout_embed in per-player mode).
        # Drop them so a strict load doesn't choke on otherwise-harmless extra keys.
        if strict:
            model_keys = set(self.state_dict().keys())
            extra = [k for k in state_dict if k not in model_keys]
            if extra:
                logger.info(
                    "Warm-start: dropping %d checkpoint param(s) absent from this model: %s",
                    len(extra),
                    sorted(extra),
                )
                for k in extra:
                    del state_dict[k]

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    CONFIG_FILENAME = LatentWorldModel.CONFIG_FILENAME

    def save_checkpoint(self, checkpoint_path: str | Path, extra_data: dict | None = None) -> None:
        if LatentWorldModel._find_config(checkpoint_path) is None:
            raise FileNotFoundError(
                f"Could not find '{LatentWorldModel.CONFIG_FILENAME}' in parent directories "
                f"of {checkpoint_path}. Make sure to save the config file alongside the checkpoint."
            )

        checkpoint = {"state_dict": self.state_dict()}
        if extra_data is not None:
            for key in extra_data.keys():
                if key in checkpoint:
                    raise ValueError(f"Key {key} already exists in checkpoint, cannot overwrite")
            checkpoint.update(extra_data)

        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str | Path, device: str | torch.device | None = None, **kwargs
    ) -> MultiWrapperWorldModel:
        from omegaconf import OmegaConf  # noqa: PLC0415 -- optional dep, used only here

        # We reuse the same config filename as the single-player model.
        config_path = LatentWorldModel._find_config(checkpoint_path)
        if config_path is None:
            raise FileNotFoundError(
                f"Could not find '{LatentWorldModel.CONFIG_FILENAME}' in parent directories "
                f"of {checkpoint_path}. Make sure to save the config file alongside the checkpoint."
            )

        config_raw = OmegaConf.load(config_path)
        raw_dict = OmegaConf.to_container(config_raw.model.architecture.config, resolve=True)
        assert isinstance(raw_dict, dict), "expected a mapping at model.architecture.config"
        config = to_namespace(raw_dict)
        model = cls(config)
        model.to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        return model
