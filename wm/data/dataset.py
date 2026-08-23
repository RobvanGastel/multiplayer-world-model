from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from wm.data.batch import VideoActionBatch


@dataclass
class ClipIndexEntry:
    path: Path
    num_frames: int  # stored frames, i.e. manifest "length" + 1 (see save_view)


def index_matches(root_dir: str | Path) -> list[ClipIndexEntry]:
    """List every game-view clip (both p0 and p1 -- video-only training doesn't
    care that only p0 has actions) across every manifest_*.jsonl shard under
    `root_dir`, without opening any .pt file: each manifest entry already
    records its clip's frame count via "length" (== stored frames - 1, see
    tools/collect_matches.py's save_view/save_game)."""
    root_dir = Path(root_dir)
    matches_dir = root_dir / "matches"
    entries: list[ClipIndexEntry] = []
    for manifest_path in sorted(root_dir.glob("manifest_*.jsonl")):
        with open(manifest_path) as f:
            for line in f:
                match = json.loads(line)
                for game in match["games"]:
                    for view in ("p0", "p1"):
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
