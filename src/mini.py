"""The 6-egg Mini: a short Dispenser in one piece, to iterate on the Lattice's look.

  uv run python src/mini.py <tag> [eggs=N] [key=value ...]   (keys: grow.Look's fields)
  uv run python src/mini.py lineup <tag> <tag> ...     the Original and built Minis in a row

Writes, in build/mini/:
- <tag>.stl          the Mini, print frame (z up)
- <tag>_use.stl      the Mini alone, use frame
- cmp_<tag>.stl      the Original and the Mini side by side, use frame (length along X, up Z),
                     for same-scale renders
- chart_<tag>.png    the unrolled skin: cells (grey lines), holes (white), material (dark)
"""

import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

import grow

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "mini"
ORIGINAL = ROOT / "reference" / "original_bio_egg_dispenser.3mf"


def use_frame(mesh, dz=0.0):
    x, y, z = mesh.vertices.T
    return trimesh.Trimesh(np.column_stack([z + dz, -x, y]), mesh.faces, process=False)


def side_by_side(mini):
    o = trimesh.load(ORIGINAL, force="mesh")
    o.vertices -= [128.0, 69.54, 0]          # centred across, on the shelf
    return trimesh.util.concatenate([use_frame(o), use_frame(mini, 270)])


def chart_png(res, path, px=3.0):
    c = res.chart
    W, H = int((c.z1 - c.z0 + 20) * px), int(c.P0 * px)
    img = Image.new("RGB", (W, H), (60, 70, 80))
    d = ImageDraw.Draw(img)
    tx = lambda z, u: ((z - c.z0 + 10) * px, (u % c.P0) * px)

    def draw(poly, **kw):
        for g in getattr(poly, "geoms", [poly]):
            if g.is_empty:
                continue
            for k in (-1, 0, 1):
                xy = [((z - c.z0 + 10) * px, (u + k * c.P0) * px) for z, u in g.exterior.coords]
                d.polygon(xy, **kw)
    for h in res.holes:
        draw(h, fill=(235, 238, 242))
    for cell in res.cells:
        draw(cell, outline=(150, 150, 150))
    img.save(path)


def lineup(tags):
    """build/mini/lineup.stl: the Original, then each Mini, along the length (use frame)."""
    o = trimesh.load(ORIGINAL, force="mesh")
    o.vertices -= [128.0, 69.54, 0]
    parts, at = [use_frame(o)], 270.0
    for t in tags:
        m = trimesh.load(OUT / f"{t}.stl")
        parts.append(use_frame(m, at))
        at += m.bounds[1][2] + 20
    trimesh.util.concatenate(parts).export(OUT / "lineup.stl")
    print(OUT / "lineup.stl")


def main():
    tag = sys.argv[1]
    if tag == "lineup":
        return lineup(sys.argv[2:])
    names = {f.name: f.type for f in fields(grow.Look)}
    kw = {}
    eggs = 6
    for a in sys.argv[2:]:
        k, v = a.split("=")
        if k == "eggs":
            eggs = int(v)
            continue
        t = names[k]
        kw[k] = int(v) if t in (int, "int") else v if t in (str, "str") else float(v)
    look = replace(grow.Look(), **kw)
    print(tag, look)
    res = grow.build(eggs=eggs, look=look)
    OUT.mkdir(parents=True, exist_ok=True)
    res.mesh.export(OUT / f"{tag}.stl")
    use_frame(res.mesh).export(OUT / f"{tag}_use.stl")
    side_by_side(res.mesh).export(OUT / f"cmp_{tag}.stl")
    chart_png(res, OUT / f"chart_{tag}.png")
    print(OUT / f"cmp_{tag}.stl")


if __name__ == "__main__":
    main()
