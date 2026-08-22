"""Shared rendering utilities used across wm/env's renderers (render_iso.py,
render_match.py) and their callers -- just the GIF-saving helper for now, so
there's one place for it instead of a near-duplicate per file."""
from __future__ import annotations


def save_gif(frames, path, fps):
    """Save a sequence of PIL Images as an animated GIF at `path`."""
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
