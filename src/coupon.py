"""The Fit coupon print job: two short Seam band blocks (one with its holes opening upward like the
Back half's, one with them on the bed like the Front half's) and four Pins.

Each block has three holes gripping at 0.03, 0.05 and 0.08 mm per side, marked by 1, 2 and 3
notches; all three open to 0.10 mm per side over their first 4 mm, so a Pin starts in easily and
lines up before it grips (joint.hole). The right one takes a firm push for the last few mm, holds
when shaken, and comes out with pliers or a twist. Its clearance becomes joint.CLEARANCE.
"""

from pathlib import Path

import joint
from plate import BED, write_project

OUT = Path(__file__).resolve().parent.parent / "exports"


def parts():
    up = joint.to_trimesh(joint.coupon_piece(holes_up=True))
    down = joint.to_trimesh(joint.coupon_piece(holes_up=False))
    pin = joint.to_trimesh(joint.pin())
    cx, cy = BED / 2, BED / 2
    up.apply_translation((cx, cy + 14, 0))
    down.apply_translation((cx, cy - 14, 0))
    out = [("block_holes_up", up), ("block_holes_on_bed", down)]
    for i in range(4):
        p = pin.copy()
        p.apply_translation((cx - 45 + 30 * i, cy - 40, 0))
        out.append((f"pin_{i + 1}", p))
    return out


if __name__ == "__main__":
    rc, info = write_project(OUT / "egg_dispenser_fit_coupon_plate.3mf", parts(), "coupon",
                             slice_check=True)
    from plate import slice_summary
    print("slice rc", rc, "time h, grams:", slice_summary(info))
