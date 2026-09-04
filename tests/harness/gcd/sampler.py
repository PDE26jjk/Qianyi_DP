"""Stratified, reproducible sampling of GarmentCodeData elements."""

from __future__ import annotations

import random
from pathlib import Path

import trimesh
import yaml


def _design_garment_type(design_params_path: Path) -> str:
    """Garment type from the upper/wb/bottom template tree (design D9)."""
    with open(design_params_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    meta = (data.get("design") or {}).get("meta") or {}
    parts = []
    for key in ("upper", "wb", "bottom"):
        value = (meta.get(key) or {}).get("v")
        if value:
            parts.append(f"{key}={value}")
    return "|".join(parts) if parts else "unknown"


def element_metadata(element_dir: Path) -> dict:
    """Return sampling metadata for one element directory."""
    element_dir = Path(element_dir)
    element_id = element_dir.name
    mesh = trimesh.load(str(element_dir / f"{element_id}_boxmesh.ply"), process=False)
    face_count = int(len(mesh.faces))
    garment_type = _design_garment_type(
        element_dir / f"{element_id}_design_params.yaml"
    )
    bucket = (
        "S" if face_count < 5000
        else "M" if face_count < 25000
        else "L" if face_count < 50000
        else "XL"
    )
    return {
        "id": element_id,
        "garment_type": garment_type,
        "face_count": face_count,
        "bucket": bucket,
    }


def scan_dataset(root: Path, scan_limit: int | None = None) -> list[dict]:
    """Scan all element directories and return their sampling metadata."""
    root = Path(root)
    entries = []
    for element_dir in sorted(root.iterdir()):
        if scan_limit is not None and len(entries) >= scan_limit:
            break
        if not element_dir.is_dir():
            continue
        try:
            entries.append(element_metadata(element_dir))
        except Exception:  # noqa: BLE001 - unreadable elements are excluded
            continue
    return entries


def select_manifest(
    root: Path,
    *,
    seed: int = 0,
    per_stratum: int = 1,
    exclude_xl: bool = True,
    max_elements: int | None = None,
    entries: list[dict] | None = None,
) -> list[dict]:
    """Select a reproducible stratified manifest.

    Strata are (garment type, size bucket); each stratum contributes
    ``per_stratum`` elements chosen with the fixed seed. XL is excluded by
    default (VRAM/time headroom, design D9/D11).
    """
    entries = entries if entries is not None else scan_dataset(root)
    rng = random.Random(seed)
    eligible = [
        e for e in entries if not (exclude_xl and e["bucket"] == "XL")
    ]
    strata: dict[tuple[str, str], list[dict]] = {}
    for entry in eligible:
        strata.setdefault((entry["garment_type"], entry["bucket"]), []).append(entry)

    selected: list[dict] = []
    for stratum, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda e: e["id"])
        rng.shuffle(ordered)
        selected.extend(ordered[:per_stratum])
    selected.sort(key=lambda e: (e["garment_type"], e["bucket"], e["id"]))
    if max_elements is not None:
        selected = selected[:max_elements]
    return selected


def write_manifest(manifest: list[dict], path: Path) -> Path:
    """Write the manifest JSON (reproducible by construction)."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
