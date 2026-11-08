"""Deliverables in exports/, rebuilt from the scripts:
  egg_dispenser_fit_coupon_plate.3mf   the Fit coupon: print and try it first
  egg_dispenser_back_plate.3mf         the Back half, on its back face, plus 6 Pins (4 + 2 spare)
  egg_dispenser_front_plate.3mf        the Front half, on its Seam face

Each is a Bambu Studio project: plate laid out, X2D settings set (plate.py), and the halves'
walls slowed above SLOW_ABOVE mm. No STEP: the geometry is mesh (ADR 0004).

v1 is Grown with seed 3. exports/archive/v1/ keeps exactly what was printed for v1.
"""

import shutil
import sys

import joint
from coupon import OUT, parts as coupon_parts
from grown import build
from halves import jobs
from plate import SLOW, SLOW_ABOVE, add_height_ranges, write_project

SEED = 3          # the chosen Grown pattern


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else None
    OUT.mkdir(exist_ok=True)
    files = [OUT / "egg_dispenser_fit_coupon_plate.3mf"]
    write_project(files[0], coupon_parts(), "coupon")
    _, halves = build(SEED)
    back, front = (joint.to_trimesh(halves[w]) for w in ("back", "front"))
    for name, parts in jobs(back, front):
        path = OUT / f"egg_dispenser_{name}_plate.3mf"
        write_project(path, parts, "half")
        add_height_ranges(path, {f"{name}_half": [(SLOW_ABOVE, 260.0, SLOW)]})
        files.append(path)
    for f in files:
        print("wrote", f)
    if version:
        archive = OUT / "archive" / version
        archive.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, archive / f.name)
        print("archived to", archive)
