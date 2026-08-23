from __future__ import annotations


def save_gif(frames, path, fps):
    """Save a sequence of PIL Images as an animated GIF at `path`.

    GIF delays are quantized to 10ms (centisecond) units, and most viewers
    (browsers, Preview, ...) treat a <=10ms per-frame delay as a legacy
    "as fast as possible" signal and clamp it to 100ms (10fps) instead --
    so anything above ~50fps silently plays back *slower*, not faster.
    Round rather than truncate, and floor the delay at 20ms (50fps) to stay
    clear of that clamp.
    """
    duration = max(20, round(1000 / fps))
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
