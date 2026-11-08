"""The Seam: the Seam band, the Pin bosses and holes on each half, the Pins, and the Fit coupon
that tests the Pin fit. One module, so the coupon tests exactly what the halves get (ADR 0002).

Manifold3d geometry in the use frame (layout.py); z is the Dispenser's length.
"""

import math

import manifold3d as mf
import numpy as np

from layout import CHANNEL, RAIL_X, SHELL_X, STRUT_R, Layout

SEGMENTS = 96

# Pin: a D-shaped rod that prints lying on its flat (ADR 0002).
PIN_R = 3.0
PIN_FLAT = 2.5            # flat face this far from the axis
PIN_LEN = 20.0
PIN_CHAMFER = 0.6
# The fit follows tea-bags-drawer-divider v3's T-pad clip, also PLA Matte on this X2D: it grips
# at 0.05 mm per side (0.1 was loose) and opens to 0.1 mm per side at its ends so it starts on
# easily. Here: a LEAD_L lead-in at LEAD_CLEARANCE finds alignment, then CLEARANCE grips.
CLEARANCE = 0.05          # per side over the hole's inner part; the Fit coupon confirms it
LEAD_CLEARANCE = 0.10     # per side over the first LEAD_L of the hole
LEAD_L = 4.0
COUPON_CLEARANCES = (0.03, 0.05, 0.08)

HOLE_DEPTH = PIN_LEN / 2 + 0.5
HOLE_CHAMFER = 0.4
BOSS_R = 6.5
BOSS_LEN = HOLE_DEPTH + 2.5
BOSS_CONE = BOSS_R - 1.0  # 45-degree cone under the boss so it prints without support

BAND_T = 5.0              # Seam band thickness along z
BAND_INNER_X = CHANNEL / 2
SPOKE_W = 4.0


def _circle(r):
    return mf.CrossSection.circle(r, SEGMENTS)


def _rect(x0, y0, x1, y1):
    return mf.CrossSection.square((x1 - x0, y1 - y0)).translate((x0, y0))


from dataclasses import dataclass


@dataclass
class SeamSpec:
    """What the Seam joint needs to know about a Dispenser's cross-section at the Seam."""
    z: float              # the Seam plane
    top: float            # top of the cage (outer)
    outer_x: float        # half the cage's outer width
    lower_rail_y: float   # Rail centre heights at the Seam
    upper_rail_y: float

    @classmethod
    def from_layout(cls, layout, z):
        return cls(z, float(layout.top_y(z)) + STRUT_R, SHELL_X + STRUT_R,
                   float(layout.lower_rail_y(z)), float(layout.upper_rail_y(z)))


def boss_centres(spec):
    """(x, y) of the four Pin bosses: the corners of the Seam band, clear of the eggs."""
    x = spec.outer_x - STRUT_R - 3.0
    return [(sx * x, y) for sx in (-1, 1) for y in (6.0, spec.top - STRUT_R - 6.0)]


def band_section(spec):
    """The Seam band's outline: a ring around the egg channel, with spokes out to the Rails."""
    top, outer_x = spec.top, spec.outer_x
    ring = _rect(-outer_x, 0, outer_x, top) - _rect(-BAND_INNER_X, 5.0, BAND_INNER_X, top - 5.0)
    yl, yu = spec.lower_rail_y, spec.upper_rail_y
    for sx in (-1, 1):
        rx = sx * RAIL_X
        ring += _rect(rx - SPOKE_W / 2, 0, rx + SPOKE_W / 2, yl)          # down to the floor
        x0, x1 = sorted((rx, sx * BAND_INNER_X))
        ring += _rect(x0, yu - SPOKE_W / 2, x1, yu + SPOKE_W / 2)                # out to the side
    return ring


def _extrude_z(section, z0, z1):
    return mf.Manifold.extrude(section, z1 - z0).translate((0, 0, z0))


def _boss(x, y, z_seam, direction):
    """A boss running from the Seam face `direction` (-1 back, +1 front) into the half, with a
    cone at its inner end."""
    if direction < 0:
        body = _extrude_z(_circle(BOSS_R), z_seam - BOSS_LEN, z_seam)
        cone = mf.Manifold.cylinder(BOSS_CONE, 1.0, BOSS_R, SEGMENTS).translate(
            (0, 0, z_seam - BOSS_LEN - BOSS_CONE))
    else:
        body = _extrude_z(_circle(BOSS_R), z_seam, z_seam + BOSS_LEN)
        cone = mf.Manifold.cylinder(BOSS_CONE, BOSS_R, 1.0, SEGMENTS).translate(
            (0, 0, z_seam + BOSS_LEN))
    return (body + cone).translate((x, y, 0))


def hole(x, y, z_face, direction, clearance=CLEARANCE):
    """A Pin hole opening at z_face and running `direction` (-1 toward -z, +1 toward +z): a
    chamfered mouth, a LEAD_L lead-in at LEAD_CLEARANCE, then `clearance` to the bottom."""
    r = PIN_R + clearance
    r_lead = PIN_R + max(LEAD_CLEARANCE, clearance)
    shaft = mf.Manifold.cylinder(HOLE_DEPTH, r, r, SEGMENTS)
    lead = mf.Manifold.cylinder(LEAD_L, r_lead, r_lead, SEGMENTS)
    flare = mf.Manifold.cylinder(HOLE_CHAMFER, r_lead + HOLE_CHAMFER, r_lead, SEGMENTS)
    h = shaft + lead + flare
    if direction < 0:
        h = h.mirror((0, 0, 1))
    return h.translate((x, y, z_face))


def seam_parts(spec, direction):
    """What the Seam adds to a half (band and bosses) and what it cuts (holes).
    direction: -1 for the Back half (it lies at z < spec.z), +1 for the Front half."""
    z_seam = spec.z
    z0, z1 = (z_seam - BAND_T, z_seam) if direction < 0 else (z_seam, z_seam + BAND_T)
    add = _extrude_z(band_section(spec), z0, z1)
    cut = mf.Manifold()
    for x, y in boss_centres(spec):
        add += _boss(x, y, z_seam, direction)
        cut += hole(x, y, z_seam, direction)
    # Trim the bosses and cones to the shelf (y >= 0).
    shelf = mf.Manifold.cube((400, 400, 1000)).translate((-200, 0, -500))
    return add ^ shelf, cut


def holes(centres, z_seam, direction, clearance=CLEARANCE):
    """Pin holes at (x, y) centres, opening at the Seam into the half on side `direction`."""
    cut = mf.Manifold()
    for x, y in centres:
        cut += hole(x, y, z_seam, direction, clearance)
    return cut


def pin():
    """One Pin lying on its flat face on z = 0, its axis along x, centred at the origin."""
    d = _circle(PIN_R) ^ _rect(-10, -PIN_FLAT, 10, 10)
    small = _circle(PIN_R - PIN_CHAMFER) ^ _rect(-10, -PIN_FLAT, 10, 10)
    core = mf.Manifold.extrude(d, PIN_LEN - 2 * PIN_CHAMFER).translate((0, 0, PIN_CHAMFER))
    ends = mf.Manifold.extrude(small, 0.01) + mf.Manifold.extrude(small, 0.01).translate(
        (0, 0, PIN_LEN - 0.01))
    p = mf.Manifold.batch_hull([core, ends])
    # axis z -> x, then the flat face (y = -PIN_FLAT) down onto z = 0
    p = p.rotate((0, 90, 0)).rotate((90, 0, 0))
    bb = p.bounding_box()
    return p.translate((-(bb[0] + bb[3]) / 2, -(bb[1] + bb[4]) / 2, -bb[2]))


# --- Fit coupon ---
COUPON_SPACING = 16.0
COUPON_W = 14.0
COUPON_H = HOLE_DEPTH + 3.0


def coupon_piece(holes_up):
    """A short block with one hole per COUPON_CLEARANCES, numbered by 1-3 notches on its face.
    holes_up: holes open at the top, as on the Back half (its Seam is the top of its print);
    otherwise at the bottom, on the bed, as on the Front half."""
    n = len(COUPON_CLEARANCES)
    w = n * COUPON_SPACING
    block = mf.Manifold.cube((w, COUPON_W, COUPON_H)).translate((-w / 2, -COUPON_W / 2, 0))
    for i, c in enumerate(COUPON_CLEARANCES):
        x = (i - (n - 1) / 2) * COUPON_SPACING
        block -= hole(x, 0, COUPON_H if holes_up else 0, -1 if holes_up else 1, clearance=c)
        for k in range(i + 1):   # notches on the -y face: 1, 2, 3 = loosest last
            nx = x + (k - i / 2) * 2.0
            block -= mf.Manifold.cube((0.8, 1.6, COUPON_H - 4)).translate(
                (nx - 0.4, -COUPON_W / 2 - 0.8, 2))
    return block


def to_trimesh(m):
    import trimesh
    mesh = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts),
                           process=False)


if __name__ == "__main__":
    L = Layout()
    spec = SeamSpec.from_layout(L, L.length / 2)
    add, cut = seam_parts(spec, -1)
    print("band+bosses volume", round(add.volume(), 1), "holes volume", round(cut.volume(), 1))
    p = pin()
    print("pin bbox", np.round(p.bounding_box(), 2), "volume", round(p.volume(), 1))
    print("bosses", boss_centres(spec))
