"""Lay a prototype's halves out as print jobs, and slice them to measure time and filament.

Both halves print with the Dispenser's length vertical: the Back half on its back face, the
Front half on its Seam face. In the use frame that is z up for both, so laying a half out is a
translation: its lowest z onto the bed, centred on the plate. The Pins ride with the Back half.
"""

import sys
from pathlib import Path

import numpy as np
import trimesh

import joint
from plate import BED, slice_summary, write_project

BUILD = Path(__file__).resolve().parent.parent / "build"


def on_bed(mesh, x=BED / 2, y=BED / 2):
    m = mesh.copy()
    lo, hi = m.bounds
    m.apply_translation((x - (lo[0] + hi[0]) / 2, y - (lo[1] + hi[1]) / 2, -lo[2]))
    return m


def jobs(back, front, pins=6):
    """[(job name, [(label, mesh)])]: the Back half with the Pins beside it, the Front half."""
    b = on_bed(back, y=BED / 2 + 20)
    pin = joint.to_trimesh(joint.pin())
    pin_parts = []
    for i in range(pins):
        p = pin.copy()
        p.apply_translation((BED / 2 - 75 + 30 * i, 22, 0))
        pin_parts.append((f"pin_{i + 1}", p))
    return [("back", [("back_half", b)] + pin_parts),
            ("front", [("front_half", on_bed(front))])]


def measure(name):
    """Slice build/<name>_back.stl and _front.stl as print jobs; print hours and grams."""
    back = trimesh.load(BUILD / f"{name}_back.stl")
    front = trimesh.load(BUILD / f"{name}_front.stl")
    total_h = total_g = 0.0
    for job, parts in jobs(back, front):
        rc, info = write_project(BUILD / f"{name}_{job}_plate.3mf", parts, "half",
                                 slice_check=True)
        h, g = slice_summary(info)
        print(f"{name} {job}: slice rc {rc}, {h:.2f} h, {g:.0f} g" if h else
              f"{name} {job}: slice failed rc {rc}")
        total_h += h or 0
        total_g += g or 0
    print(f"{name} total: {total_h:.1f} h, {total_g:.0f} g")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        measure(n)
