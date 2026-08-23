"""WASD key-press encoding for continuous 2D mallet actions.

Lets a recorded (ax, ay) action also be expressed in the same discrete
W/A/S/D vocabulary a human would drive interactively with, so recorded data
and a future keyboard-interactive mode share one action representation.
Convention (ax>0->D, ax<0->A, ay>0->S, ay<0->W) matches the original
wm/env/play_with_keys.py's action_to_keys() before it was removed.
"""
from __future__ import annotations

import numpy as np

WASD_KEYS = ("w", "a", "s", "d")


def action_to_wasd(ax: float, ay: float, thresh: float = 0.15) -> tuple[bool, bool, bool, bool]:
    """(ax, ay) -> (w, a, s, d) key-press booleans."""
    return (ay < -thresh, ax < -thresh, ay > thresh, ax > thresh)


def actions_to_wasd(actions: np.ndarray, thresh: float = 0.15) -> np.ndarray:
    """(T, 2) continuous actions -> (T, 4) multi-hot W/A/S/D array, int8."""
    ax, ay = actions[:, 0], actions[:, 1]
    w, a, s, d = ay < -thresh, ax < -thresh, ay > thresh, ax > thresh
    return np.stack([w, a, s, d], axis=-1).astype(np.int8)
