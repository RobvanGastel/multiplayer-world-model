from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from wm.data.batch import VideoActionBatch
from wm.world_model.action_configs import ActionTensors


@dataclass
class ClipIndexEntry:
    path: Path
    num_frames: int  # stored frames, i.e. manifest "length" + 1 (see save_view)


def index_matches(root_dir: str | Path, views: tuple[str, ...] = ("p0", "p1")) -> list[ClipIndexEntry]:
    """List every game-view clip in `views` across every manifest_*.jsonl shard under `root_dir`,
    without opening any .pt file: each manifest entry already records its clip's frame count via
    "length" (== stored frames - 1, see tools/data/collect_matches.py's save_view/save_game).

    Video-only training doesn't care that only p0 ("striker") has recorded actions -- p1
    ("defender") is always saved with actions=None -- so create_dataloader takes the default
    views=("p0", "p1"); create_loader passes views=("p0",) since it needs actions."""
    root_dir = Path(root_dir)
    matches_dir = root_dir / "matches"
    entries: list[ClipIndexEntry] = []
    for manifest_path in sorted(root_dir.glob("manifest_*.jsonl")):
        with open(manifest_path) as f:
            for line in f:
                match = json.loads(line)
                for game in match["games"]:
                    for view in views:
                        clip = game[view]
                        entries.append(ClipIndexEntry(
                            path=matches_dir / clip["file"],
                            num_frames=clip["length"] + 1,
                        ))
    return entries


class MatchClipDataset(Dataset):
    """Samples a random fixed-length video window from each collected match clip.

    Clips shorter than `clip_len` (e.g. quick own-goals) can't fill a window and
    are dropped at index time.
    """

    def __init__(self, root_dir: str | Path, clip_len: int) -> None:
        self.clip_len = clip_len
        all_entries = index_matches(root_dir)
        self.entries = [e for e in all_entries if e.num_frames >= clip_len]
        if not self.entries:
            raise ValueError(
                f"No clips with >= {clip_len} frames found under {root_dir} "
                f"({len(all_entries)} clips indexed total)"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tensor:
        entry = self.entries[idx]
        # Our own files, but torch>=2.6 defaults weights_only=True which would
        # reject the plain str fields (outcome/view/player/role) alongside frames.
        frames = torch.load(entry.path, map_location="cpu", weights_only=False)["frames"]
        start = random.randint(0, frames.shape[0] - self.clip_len)
        return frames[start:start + self.clip_len]


def collate_video(batch: list[Tensor]) -> VideoActionBatch:
    return VideoActionBatch(video=torch.stack(batch))


def _worker_init_fn(worker_id: int) -> None:
    # DataLoader forks worker processes, which inherit the parent's `random`
    # module state as-is -- without this every worker would sample the exact
    # same sequence of clip-window offsets. torch.initial_seed() is already
    # per-worker (DataLoader sets it before running this fn), so reseed
    # stdlib `random` from it too.
    random.seed(torch.initial_seed() % (2 ** 32))


def create_dataloader(
    root_dir: str | Path,
    clip_len: int,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    dataset = MatchClipDataset(root_dir, clip_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_video,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        drop_last=True,
        pin_memory=True,
    )


class ActionClipDataset(Dataset):
    """Samples a random fixed-length (frames, actions_wasd) window from each p0 ("striker") match
    clip -- MatchClipDataset's action-conditioned counterpart, for the world model.

    Only p0 clips carry actions (see tools/data/collect_matches.py's save_game: p1/"defender" is
    always saved with actions=None), so p1 clips are excluded entirely rather than handled as a
    missing-actions case.
    """

    def __init__(self, root_dir: str | Path, clip_len: int) -> None:
        self.clip_len = clip_len
        all_entries = index_matches(root_dir, views=("p0",))
        self.entries = [e for e in all_entries if e.num_frames >= clip_len]
        if not self.entries:
            raise ValueError(
                f"No p0 clips with >= {clip_len} frames found under {root_dir} "
                f"({len(all_entries)} p0 clips indexed total)"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        entry = self.entries[idx]
        payload = torch.load(entry.path, map_location="cpu", weights_only=False)
        frames = payload["frames"]
        # actions_wasd is one shorter than frames (frame[t]/action[t] together explain the
        # transition to frame[t+1] -- see collect_matches.py's save_game), so a clip_len-frame
        # window only has clip_len - 1 real actions.
        actions_wasd = payload["actions_wasd"]
        start = random.randint(0, frames.shape[0] - self.clip_len)
        return frames[start:start + self.clip_len], actions_wasd[start:start + self.clip_len - 1]


def collate_action(batch: list[tuple[Tensor, Tensor]], actions_config) -> VideoActionBatch:
    frames, actions_wasd = zip(*batch)
    key_presses = torch.stack(actions_wasd).to(torch.int32)

    action_tensors = ActionTensors(config=actions_config, batch_size=len(batch))
    action_tensors.key_presses = key_presses
    action_tensors.mouse_movements = torch.zeros((len(batch), key_presses.shape[1], 2), dtype=torch.float32)
    # game_mouse_sensitivity stays all-NaN (ActionTensors' default) -- this dataset is
    # keyboard-only, matching the encoder's documented "no mouse" convention.
    return VideoActionBatch(video=torch.stack(frames), actions=action_tensors)


def create_loader(
    root_dir: str | Path,
    clip_len: int,
    batch_size: int,
    # a plain object with .valid_keys (must be ("w", "a", "s", "d"), the only keys
    # tools/data/collect_matches.py's actions_to_wasd produces) -- e.g. model.config.actions,
    # loaded from configs/world_model/latent_world_model.yml.
    actions_config,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader:
    dataset = ActionClipDataset(root_dir, clip_len)
    generator = None
    if seed is not None:
        # Fixed generator (rather than global torch seeding) so the val/metrics loaders sample the
        # same held-out subsample on every eval regardless of train-loop RNG state at call time.
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_action, actions_config=actions_config),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=generator,
        drop_last=True,
        pin_memory=True,
    )
