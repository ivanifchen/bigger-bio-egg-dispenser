"""Stretch: the Original's own mesh, made 16 eggs long.

1. Into the use frame (x across, y up from the shelf, z from the back face).
2. Stretch the straight middle (z 70-210, where the Rails run straight) so the Dispenser is as
   long as Grown; the Turn and the Exit cradle keep their shape.
3. Bend it back to SLOPE_DEG on both Levels: stretching flattens them. Everything above the
   floor rises toward the back by LOWER_LIFT (steepening the Lower level), and everything above
   the Lower level's eggs rises toward the front by UPPER_LIFT (steepening the Upper level).
   The bends fade in smoothly, so the Lattice just leans.
4. Add continuous Side guides along both Levels at egg-centre height (ADR 0003).
5. Add V ties across the top and bottom, as Grown's, so the sides never stand free more than
   about 40 mm while printing (the Original stands free up to 130 mm; stretched, 170).
6. Split at mid-length and add the Seam joint (joint.py), as Grown.

Reads reference/original_bio_egg_dispenser.3mf, which never leaves this machine (see .gitignore).
"""

import math
from pathlib import Path

import manifold3d as mf
import numpy as np
import trimesh

import joint
from layout import GUIDE_R, GUIDE_X, LARGE, Layout, centre_offset

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "reference" / "original_bio_egg_dispenser.3mf"
BUILD = ROOT / "build"

Z1, Z2 = 70.0, 210.0       # the straight middle that stretches (Original's z)
SLOPE_DEG = 2.2


def original_use_frame():
    m = trimesh.load(ORIGINAL, force="mesh")
    lo, hi = m.bounds
    x, y, z = m.vertices.T
    v = np.column_stack([x - (lo[0] + hi[0]) / 2, hi[1] - y, z - lo[2]])   # Y flips: shelf at 0
    m = trimesh.Trimesh(v, m.faces[:, ::-1], process=True)                # undo the mirror
    assert m.volume > 0
    return m


def rail_tracks(m, step=4.0):
    """Centre lines of the Original's four Rails, (z, y) per level, from cross-sections."""
    lower, upper = [], []
    for z in np.arange(4, m.bounds[1][2] - 4, step):
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            continue
        p2, T = sec.to_2D()
        for pg in p2.polygons_full:
            c = trimesh.transform_points([[*pg.centroid.coords[0], 0]], T)[0]
            if 11 < abs(c[0]) < 15.5 and pg.area < 30:
                (lower if c[1] < 35 else upper).append((z, c[1]))
    return np.array(lower), np.array(upper)


def stretch_z(z, extra):
    z = np.asarray(z, float)
    f = (Z2 - Z1 + extra) / (Z2 - Z1)
    return np.where(z <= Z1, z, np.where(z >= Z2, z + extra, Z1 + (z - Z1) * f))


def smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def build(write=True):
    L = Layout()
    m = original_use_frame()
    lower, upper = rail_tracks(m)
    length0 = m.bounds[1][2]
    extra = L.length - length0
    stretched = (Z2 - Z1 + extra)
    t = math.tan(math.radians(SLOPE_DEG))

    def straight_slope(track):
        sel = (track[:, 0] > Z1 + 5) & (track[:, 0] < Z2 - 5)
        return np.polyfit(track[sel, 0], track[sel, 1], 1)[0]
    s_low, s_up = straight_slope(lower), straight_slope(upper)
    # After stretching, each slope shrinks by the stretch factor; the lifts make up the rest.
    f = stretched / (Z2 - Z1)
    lower_lift = stretched * (t - (-s_low) / f)
    upper_lift = stretched * (t - s_up / f) + lower_lift
    print(f"Original slopes: Lower {math.degrees(math.atan(-s_low)):.2f} deg, "
          f"Upper {math.degrees(math.atan(s_up)):.2f} deg; stretch x{f:.2f}; "
          f"lifts: Lower back {lower_lift:.1f} mm, Upper front {upper_lift:.1f} mm")

    def warp(points):
        x, y, z = np.asarray(points, float).T
        s = np.clip((z - Z1) / (Z2 - Z1), 0, 1)
        # Lower lift: everything above the floor, most at the back, none at the front.
        # Upper lift: everything above the Lower level's eggs, most at the front; it includes
        # the Lower lift it has to undo at the back.
        dy = lower_lift * (1 - s) * smoothstep(y, 1.0, 6.0) \
            + upper_lift * s * smoothstep(y, 30.0, 55.0)
        return np.column_stack([x, y + dy, stretch_z(z, extra)])
    v = warp(m.vertices)
    m2 = trimesh.Trimesh(v, m.faces, process=False)
    body = mf.Manifold(mf.Mesh(vert_properties=np.asarray(m2.vertices, np.float32),
                               tri_verts=np.asarray(m2.faces, np.uint32)))
    assert body.status() == mf.Error.NoError, body.status()

    # Rail centre lines after the warp, and the egg-centre lines above them.
    low_w = warp(np.column_stack([np.full(len(lower), 13.0), lower[:, 1], lower[:, 0]]))
    up_w = warp(np.column_stack([np.full(len(upper), 13.0), upper[:, 1], upper[:, 0]]))
    c_off = centre_offset(LARGE)

    def level_line(track_w, z_from, z_to):
        sel = (track_w[:, 2] >= z_from) & (track_w[:, 2] <= z_to)
        tw = track_w[sel]
        order = np.argsort(tw[:, 2])
        return tw[order][:, 2], tw[order][:, 1] + c_off
    zl, yl = level_line(low_w, 30, L.length - 30)
    zu, yu = level_line(up_w, 50, L.length - 30)
    slope_l = math.degrees(math.atan(-np.polyfit(zl[(zl > 80) & (zl < 330)],
                                                 yl[(zl > 80) & (zl < 330)], 1)[0]))
    slope_u = math.degrees(math.atan(np.polyfit(zu[(zu > 80) & (zu < 330)],
                                                yu[(zu > 80) & (zu < 330)], 1)[0]))
    print(f"after: Lower {slope_l:.2f} deg, Upper {slope_u:.2f} deg")

    # Side guides: capsules along both egg-centre lines, both sides.
    guides = mf.Manifold()
    for zs_, ys_ in ((zl, yl), (zu, yu)):
        zz = np.arange(zs_[0], zs_[-1], 6.0)
        yy = np.interp(zz, zs_, ys_)
        for sx in (-1, 1):
            pts = [(sx * GUIDE_X, y, z) for y, z in zip(yy, zz)]
            for a, b in zip(pts[:-1], pts[1:]):
                guides += mf.Manifold.batch_hull([
                    mf.Manifold.sphere(GUIDE_R, 24).translate(a),
                    mf.Manifold.sphere(GUIDE_R, 24).translate(b)])
    body = body + guides

    # V ties across the top and bottom (grown.chevrons), clear of this cage's eggs.
    import grown
    from layout import BOTTOM_Y, STRUT_R, XL
    lift_xl = centre_offset(XL) - c_off
    path_l = np.column_stack([np.interp(np.arange(zl[0], zl[-1], 2), zl, yl),
                              np.arange(zl[0], zl[-1], 2)])
    path_u = np.column_stack([np.interp(np.arange(zu[0], zu[-1], 2), zu, yu),
                              np.arange(zu[0], zu[-1], 2)])
    path = np.concatenate([path_l, path_u])
    eggs = [(path, LARGE["length"] / 2, LARGE["diameter"] / 2),
            (path + [lift_xl, 0], XL["length"] / 2, XL["diameter"] / 2)]
    zv, yv = m2.vertices[:, 2], m2.vertices[:, 1]
    zbins = np.arange(0, L.length + 4, 4.0)
    tops = np.array([yv[(zv >= a) & (zv < a + 4)].max() if ((zv >= a) & (zv < a + 4)).any()
                     else np.nan for a in zbins])
    top_at = lambda z: float(np.interp(z, zbins + 2, tops)) - STRUT_R
    g = grown.Graph()
    rng = np.random.default_rng(7)
    grown.chevrons(g, lambda q: np.array([q[1], top_at(q[0]), q[0]]), 8.0, L.length - 95.0,
                   rng, eggs, "top")
    grown.chevrons(g, lambda q: np.array([q[1], BOTTOM_Y, q[0]]), 8.0, L.length - 80.0,
                   rng, eggs, "bottom")
    A, B, R, K = g.arrays()
    ties = mf.Manifold()
    for a, b, r in zip(A, B, R):
        ties += mf.Manifold.batch_hull([mf.Manifold.sphere(r, 20).translate(a),
                                        mf.Manifold.sphere(r, 20).translate(b)])
    for c, r in g.nodes:
        ties += mf.Manifold.sphere(r, 20).translate(c)
    shelf = mf.Manifold.cube((400, 400, 1000)).translate((-200, 0, -500))
    body = body + (ties ^ shelf)
    print(f"  ties: {len(A)} Strut segments")

    # The Seam: halfway, with the band fitted to this cage's cross-section there.
    zs = L.length / 2
    bb = body.bounding_box()
    sec_top = max(p[1] for p in m2.vertices[np.abs(m2.vertices[:, 2] - zs) < 3])
    spec = joint.SeamSpec(zs, sec_top, max(abs(bb[0]), abs(bb[3])) - 0.5,
                          float(np.interp(zs, low_w[:, 2], low_w[:, 1])),
                          float(np.interp(zs, up_w[np.argsort(up_w[:, 2])][:, 2],
                                          up_w[np.argsort(up_w[:, 2])][:, 1])))
    big = 1000.0
    halves = {}
    for which, direction in (("back", -1), ("front", 1)):
        keep = mf.Manifold.cube((big, big, big)).translate(
            (-big / 2, -big / 2, zs - big if direction < 0 else zs))
        add, cut = joint.seam_parts(spec, direction)
        half = ((body ^ keep) + add) - cut
        halves[which] = half
        print(f"  {which}: {half.num_tri()} triangles, {half.volume() / 1000:.1f} cm3")
        if write:
            BUILD.mkdir(exist_ok=True)
            joint.to_trimesh(half).export(BUILD / f"stretch_{which}.stl")
    if write:
        joint.to_trimesh(halves["back"] + halves["front"]).export(BUILD / "stretch.stl")
    return halves


if __name__ == "__main__":
    build()
