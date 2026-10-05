"""The action encoder: embeds keyboard actions into per-latent-frame conditioning tokens.

The encoder embeds each key with its own learned embedding, temporally pools the key presses of each
latent frame with a learned linear layer, and prepends a learned initial-action token so the first
latent frame has a conditioning token.

The mouse branch is inherited from MIRA (Rocket League is played with keyboard and mouse). This game
is keyboard-only, so it always sees zero mouse movement and an unknown (NaN) mouse sensitivity, and
carries no information. It is kept because its weights are part of the trained checkpoint and it
takes half of the embedding width.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from wm.training.weights import init_weights
from wm.world_model.action_configs import ActionTensors
from wm.world_model.schedule import symlog_normalize


class ActionEncoder(torch.nn.Module):
    def __init__(
        self,
        num_key_presses: int,
        dim: int,
        temporal_downsampling: int,
        dropout_prob: float = 0.0,
        max_mouse_movement: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.temporal_downsampling = temporal_downsampling
        self.dropout_prob = dropout_prob
        self.max_mouse_movement = max_mouse_movement

        mouse_dim = dim // 2
        keyboard_dim = dim - mouse_dim
        self.mouse_mlp = nn.Linear(2, mouse_dim)
        self.mouse_sensitivity_mlp = nn.Linear(1, mouse_dim)
        self.mouse_sensitivity_dropout_token = nn.Parameter(0.02 * torch.randn(1, 1, mouse_dim))

        # closest power of 2
        keyboard_split_dim = 2 ** math.floor(math.log2(keyboard_dim / num_key_presses))
        keyboard_remaining_dim = keyboard_dim - num_key_presses * keyboard_split_dim

        if keyboard_remaining_dim > 0:
            self.register_buffer(
                "keyboard_zero_vector",
                torch.zeros((1, 1, keyboard_remaining_dim)),
                persistent=False,
            )
        else:
            self.keyboard_zero_vector = None

        self.keyboard_embedding_dict = nn.ModuleDict()
        for k in range(num_key_presses):
            self.keyboard_embedding_dict[str(k)] = nn.Embedding(2, keyboard_split_dim)

        self.keyboard_mlp = nn.Linear(keyboard_dim, keyboard_dim)

        # Per-player action dropout (classifier-free style): a dropped player row has every key
        # replaced by a learned per-key token, and its mouse slot by mouse_dropout_token.
        self.key_dropout_embed = None
        if dropout_prob > 0:
            self.key_dropout_embed = nn.Parameter(0.02 * torch.randn(num_key_presses, keyboard_split_dim))

        # Learned pooling of the temporal_downsampling key presses that belong to one latent frame.
        self.mouse_temporal_pool = nn.Linear(temporal_downsampling * mouse_dim, mouse_dim)
        self.keyboard_temporal_pool = nn.Linear(temporal_downsampling * keyboard_dim, keyboard_dim)

        self.joint_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.mouse_dropout_token = None
        if dropout_prob > 0:
            self.mouse_dropout_token = nn.Parameter(0.02 * torch.randn(1, 1, mouse_dim))

        # initial action token
        self.initial_action_token = nn.Parameter(0.02 * torch.randn(1, 1, dim))

        self.apply(init_weights)

    def _sample_drop_mask(self, batch_size: int, device) -> Tensor:
        """Per-(player-)row action dropout: each row is dropped w.p. ``dropout_prob``. Since
        MultiWrapper feeds a (b*n_players) batch, per-row == per-player. Returns (b,) bool."""
        return torch.rand(batch_size, device=device) < self.dropout_prob

    def forward(self, actions: ActionTensors, drop_mask: Tensor | None = None) -> Tensor:
        mouse_movements = actions.mouse_movements
        batch_size, n_actions, _ = mouse_movements.shape
        device = mouse_movements.device

        # we record raw mouse deltas Δx and Δy, their unit is in dots through the formula
        # Δx = physical_mouse_displacement_in_inches * DPI , where DPI is mouse-dependent.
        # This translates to an in-game movement (Δx, Δy) * game_mouse_sensitivity
        mouse_movements = mouse_movements.clamp(-self.max_mouse_movement, self.max_mouse_movement)
        delta_xy_normalized = symlog_normalize(mouse_movements, scale=1.0, max_value=self.max_mouse_movement)
        mouse_embed = self.mouse_mlp(delta_xy_normalized)

        # mouse sensitivity
        mouse_sensitivity = rearrange(actions.game_mouse_sensitivity, "b -> b 1 1")
        mouse_sensitivity_mask = torch.isnan(mouse_sensitivity)
        mouse_sensitivity = torch.nan_to_num(mouse_sensitivity, nan=1.0)
        mouse_sensitivity_embed = self.mouse_sensitivity_mlp(mouse_sensitivity)
        mouse_sensitivity_embed = torch.where(
            mouse_sensitivity_mask, self.mouse_sensitivity_dropout_token, mouse_sensitivity_embed
        )

        mouse_embed = mouse_embed + mouse_sensitivity_embed

        key_presses = actions.key_presses

        # Per-(player-)row action dropout: sampled in training; at inference a (b,) drop_mask drops
        # all of a row's actions, a (b, num_keys) one only the selected keys.
        key_drop_mask, mouse_drop_mask = None, None
        if self.key_dropout_embed is not None:
            if self.training:
                mouse_drop_mask = self._sample_drop_mask(batch_size, device)
            elif drop_mask is not None and drop_mask.dim() == 1:
                mouse_drop_mask = drop_mask
            elif drop_mask is not None:
                key_drop_mask = drop_mask
            if mouse_drop_mask is not None:
                key_drop_mask = mouse_drop_mask.unsqueeze(-1).expand(-1, key_presses.shape[-1])

        keyboard_embed_list = []
        for k in range(key_presses.shape[-1]):
            embed_k = self.keyboard_embedding_dict[str(k)](key_presses[:, :, k])
            if key_drop_mask is not None:
                assert self.key_dropout_embed is not None
                token = self.key_dropout_embed[k].to(embed_k.dtype)
                embed_k = torch.where(key_drop_mask[:, k].view(-1, 1, 1), token, embed_k)
            keyboard_embed_list.append(embed_k)

        if self.keyboard_zero_vector is not None:
            keyboard_embed_list.append(self.keyboard_zero_vector.expand(batch_size, n_actions, -1))

        keyboard_embed = torch.cat(keyboard_embed_list, dim=-1)
        keyboard_embed = self.keyboard_mlp(keyboard_embed)

        # Temporally downsample
        mouse_embed = mouse_embed.unflatten(dim=1, sizes=(-1, self.temporal_downsampling))
        keyboard_embed = keyboard_embed.unflatten(dim=1, sizes=(-1, self.temporal_downsampling))
        mouse_embed = self.mouse_temporal_pool(mouse_embed.flatten(2))
        keyboard_embed = self.keyboard_temporal_pool(keyboard_embed.flatten(2))

        # The keys were dropped per key at the embedding level above; drop the mouse slot with them.
        if mouse_drop_mask is not None:
            mouse_dropout_token = self.mouse_dropout_token.to(mouse_embed.dtype)
            mouse_embed = torch.where(mouse_drop_mask.view(-1, 1, 1), mouse_dropout_token, mouse_embed)

        actions_embed = torch.cat((mouse_embed, keyboard_embed), dim=-1)
        actions_embed = self.joint_mlp(actions_embed)

        # append initial action token
        initial_action_token = self.initial_action_token.expand(batch_size, -1, -1)
        actions_embed = torch.cat([initial_action_token, actions_embed], dim=1)
        return actions_embed
