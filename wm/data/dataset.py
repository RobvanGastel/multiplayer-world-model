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
from wm.world_model.action_configs import ActionTensors, stack_action_tensors


@dataclass
class ClipIndexEntry:
    path: Path
    num_frames: int  # stored frames, i.e. manifest "length" + 1 (see save_view)


def index_matches(root_dir: str | Path, views: tuple[str, ...] = ("p0", "p1")) -> list[ClipIndexEntry]:
    """List every game-view clip in `views` across every manifest_*.jsonl shard under `root_dir`,
    without opening any .pt file: each manifest entry already records its clip's frame count via
    "length" (== stored frames - 1, see tools/agent/collect_matches.py's save_view/save_game).

    Both create_dataloader (video-only) and create_loader (action-conditioned) use the default
    views=("p0", "p1") -- both players' clips carry real actions now (see ActionClipDataset)."""
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
        frames = torch.load(entry.path, map_location="cpu", weights_only=False, mmap=True)["frames"]
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
    """Samples a random fixed-length (frames, actions_wasd) window from each match clip, from
    either player's perspective -- MatchClipDataset's action-conditioned counterpart, for the
    world model.

    Both p0 ("striker") and p1 ("defender") clips carry real actions (p1's is the opponent
    policy's mirrored-frame action, see AirHockeyMatchEnv.step's `defender_action` and
    tools/agent/collect_matches.py's save_game), so both are included as independent examples --
    each already has its own view-consistent (frames, actions) pairing, no player-identity
    conditioning needed to tell them apart.
    """

    def __init__(self, root_dir: str | Path, clip_len: int) -> None:
        self.clip_len = clip_len
        all_entries = index_matches(root_dir, views=("p0", "p1"))
        self.entries = [e for e in all_entries if e.num_frames >= clip_len]
        if not self.entries:
            raise ValueError(
                f"No clips with >= {clip_len} frames found under {root_dir} "
                f"({len(all_entries)} clips indexed total)"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        entry = self.entries[idx]
        payload = torch.load(entry.path, map_location="cpu", weights_only=False, mmap=True)
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
    # tools/agent/collect_matches.py's actions_to_wasd produces) -- e.g. model.config.actions,
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


@dataclass
class GameIndexEntry:
    path_p0: Path
    path_p1: Path
    num_frames: int  # shared by both views (see collect_matches.py's save_game)


def index_games(root_dir: str | Path) -> list[GameIndexEntry]:
    """List every collected game under `root_dir`, keeping its p0/p1 clips paired (unlike
    index_matches, which flattens both views into independent entries) -- for
    MultiPlayerActionClipDataset, which needs both players' clips from the SAME game to build one
    multiplayer training sample."""
    root_dir = Path(root_dir)
    matches_dir = root_dir / "matches"
    entries: list[GameIndexEntry] = []
    for manifest_path in sorted(root_dir.glob("manifest_*.jsonl")):
        with open(manifest_path) as f:
            for line in f:
                match = json.loads(line)
                for game in match["games"]:
                    entries.append(GameIndexEntry(
                        path_p0=matches_dir / game["p0"]["file"],
                        path_p1=matches_dir / game["p1"]["file"],
                        num_frames=game["p0"]["length"] + 1,
                    ))
    return entries


class MultiPlayerActionClipDataset(Dataset):
    """Samples a random fixed-length (frames, actions_wasd) window from each collected game, for
    BOTH players at once -- MultiWrapperWorldModel's counterpart to ActionClipDataset, which only
    ever returns one player's view per sample.

    Both players' clips of a game share the same frame count (see index_games -- both p0/p1
    lengths come from collect_matches.py's save_game, which pads both views identically), so a
    single random crop offset applies to both -- p0 and p1 frame `t` always depict the same
    simulation timestep.
    """

    def __init__(self, root_dir: str | Path, clip_len: int) -> None:
        self.clip_len = clip_len
        all_entries = index_games(root_dir)
        self.entries = [e for e in all_entries if e.num_frames >= clip_len]
        if not self.entries:
            raise ValueError(
                f"No games with >= {clip_len} frames found under {root_dir} "
                f"({len(all_entries)} games indexed total)"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> list[tuple[Tensor, Tensor]]:
        entry = self.entries[idx]
        payload_p0 = torch.load(entry.path_p0, map_location="cpu", weights_only=False, mmap=True)
        payload_p1 = torch.load(entry.path_p1, map_location="cpu", weights_only=False, mmap=True)
        start = random.randint(0, payload_p0["frames"].shape[0] - self.clip_len)
        end = start + self.clip_len
        # Player-ordered: [p0, p1] -- MultiWrapperWorldModel.forward's
        # rearrange("(b p) ... -> b p ...") relies on this exact ordering within each game.
        return [
            (payload_p0["frames"][start:end], payload_p0["actions_wasd"][start:end - 1]),
            (payload_p1["frames"][start:end], payload_p1["actions_wasd"][start:end - 1]),
        ]


def collate_multiplayer_action(
    batch: list[list[tuple[Tensor, Tensor]]], actions_config
) -> VideoActionBatch:
    """Flatten each game's [(frames_p0, actions_p0), (frames_p1, actions_p1)] into one batch with
    players contiguous per game -- [game0_p0, game0_p1, game1_p0, game1_p1, ...] -- matching
    MultiWrapperWorldModel.forward's grouping invariant. Reuses collate_action for the actual
    stacking once flattened, since each row is the same (frames, actions_wasd) tuple shape."""
    rows = [row for game in batch for row in game]
    return collate_action(rows, actions_config=actions_config)


def create_multiplayer_loader(
    root_dir: str | Path,
    clip_len: int,
    batch_size: int,
    # a plain object with .valid_keys, e.g. model.config.actions -- see create_loader.
    actions_config,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """MultiWrapperWorldModel's counterpart to create_loader: each dataset item is one game (both
    players -- MultiPlayerActionClipDataset is p0/p1-specific, matching this repo's 2-player
    collection), and `batch_size` counts whole games -- not rows -- so the loader's actual batch
    dimension is `batch_size * 2`, matching MultiWrapperWorldModel(n_players=2)'s expected
    grouping."""
    dataset = MultiPlayerActionClipDataset(root_dir, clip_len)
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
        collate_fn=partial(collate_multiplayer_action, actions_config=actions_config),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=generator,
        drop_last=True,
        pin_memory=True,
    )


class LatentMultiPlayerDataset(Dataset):
    """MultiPlayerActionClipDataset over pre-encoded codec latents (tools/data/encode_latents.py).
    The whole split is held in RAM (a few GB), so a sample is a slice rather than a file read and
    the frozen codec never runs during training.

    Latent crops start at any latent index s, i.e. video frame s * td -- the codec's temporal conv
    has kernel == stride == td, so a full-game encoding sliced there equals encoding that window
    (encode_latents.py checks this). Actions keep the video loader's convention of one fewer than
    the window's frames."""

    def __init__(self, latents_dirs: list[str | Path], n_latents: int) -> None:
        self.n_latents = n_latents
        games = []
        for latents_dir in latents_dirs:
            shards = sorted(Path(latents_dir).glob("*.pt"))
            if not shards:
                raise ValueError(f"No latent shards under {latents_dir} -- run tools/data/encode_latents.py")
            for shard in shards:
                payload = torch.load(shard, map_location="cpu", weights_only=False)
                self.temporal_downsampling = payload["temporal_downsampling"]
                codec = payload["codec_checkpoint"]
                assert getattr(self, "codec_checkpoint", codec) == codec, f"{shard}: mixed codecs"
                self.codec_checkpoint = codec
                games += payload["games"]
        self.games = [g for g in games if g["z_p0"].shape[0] >= n_latents]

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, idx: int) -> list[tuple[Tensor, Tensor]]:
        game = self.games[idx]
        start = random.randint(0, game["z_p0"].shape[0] - self.n_latents)
        f0, n_actions = start * self.temporal_downsampling, self.n_latents * self.temporal_downsampling - 1
        return [
            (game[f"z_{p}"][start:start + self.n_latents], game[f"actions_{p}"][f0:f0 + n_actions])
            for p in ("p0", "p1")
        ]


def collate_multiplayer_latents(
    batch: list[list[tuple[Tensor, Tensor]]], actions_config
) -> VideoActionBatch:
    """collate_multiplayer_action for LatentMultiPlayerDataset: the stacked tensor lands in
    `.latents` instead of `.video`."""
    collated = collate_multiplayer_action(batch, actions_config)
    return VideoActionBatch(video=None, actions=collated.actions, latents=collated.video)


def create_latent_multiplayer_loader(
    # one collected-data dir, or a list of them to train on together (e.g. old + new collections)
    root_dir: str | Path | list[str | Path],
    n_latents: int,
    batch_size: int,
    actions_config,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """create_multiplayer_loader over `<root>/latents` of each root (see LatentMultiPlayerDataset)."""
    roots = root_dir if isinstance(root_dir, list) else [root_dir]
    dataset = LatentMultiPlayerDataset([Path(r) / "latents" for r in roots], n_latents)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_multiplayer_latents, actions_config=actions_config),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=generator,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
