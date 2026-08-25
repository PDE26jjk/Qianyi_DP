"""Optional per-frame rendering and GIF assembly for simulation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def render_gif(
    frames: np.ndarray,
    triangles: np.ndarray,
    output_path: str | Path,
    *,
    duration_ms: int = 100,
    z_pad: float = 0.05,
    frame_skip: int = 1,
    dpi: int = 90,
) -> Path:
    """Render per-frame cloth surfaces and assemble them into a GIF.

    Uses matplotlib's ``plot_trisurf`` per frame and Pillow for GIF assembly.
    ``frame_skip`` renders every Nth frame to keep the artifact small.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    triangles = np.asarray(triangles, dtype=np.int64)
    frames = frames[:: max(1, frame_skip)]

    z_min = float(np.min(frames[:, :, 2])) - z_pad
    z_max = float(np.max(frames[:, :, 2])) + z_pad
    images: list[Image.Image] = []
    for frame_index, frame in enumerate(frames):
        x = frame[:, 0]
        y = frame[:, 1]
        z = frame[:, 2]
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(x, y, z, triangles=triangles, cmap="viridis", linewidth=0)
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
        ax.set_zlim(z_min, z_max)
        ax.set_title(f"frame {frame_index * max(1, frame_skip)}")
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        image = Image.fromarray(buffer[:, :, :3])
        images.append(image)
        plt.close(fig)

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path


def write_gallery(gallery_path: str | Path, entries: list[dict]) -> Path:
    """Write a gallery.md summary index for generated GIF artifacts."""
    gallery_path = Path(gallery_path)
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Simulation Gallery", ""]
    for entry in entries:
        lines.append(f"## {entry['name']}")
        lines.append("")
        lines.append(f"- gif: {entry['gif']}")
        if entry.get("params"):
            lines.append(f"- params: {entry['params']}")
        lines.append("")
    gallery_path.write_text("\n".join(lines), encoding="utf-8")
    return gallery_path
