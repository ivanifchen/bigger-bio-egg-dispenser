"""Measure a prototype's halves against the rules agreed for the Dispenser.

  uv run python src/check.py grown_1 stretch

For each build/<name>_back.stl and _front.stl (print frame: z up, the Dispenser's length):
- size: length, height, width, and the 406 mm Shelf depth
- overhang: surface facing down more than 45 degrees, as a share of all surface (the Original:
  5.3% above its first layer)
- ties: at every print height, how far the left and right sides stand free (agreed: <= 40
  mm, plus a little for the 45-degree legs)
- islands: separate loops per layer, which drive print time (the Original: 13)
- egg clearance: the swept XL and Large eggs against everything but the Rails and Side guides
  (Grown only: needs the graph)
"""

import sys
from pathlib import Path

import manifold3d as mf
import numpy as np
import trimesh

from layout import Layout

BUILD = Path(__file__).resolve().parent.parent / "build"
SHELF_DEPTH = 406.0


def load(name, which):
    return trimesh.load(BUILD / f"{name}_{which}.stl")


def overhang_share(m, bed_z):
    n, a, c = m.face_normals, m.area_faces, m.triangles_center
    down = (n[:, 2] < -np.cos(np.radians(45))) & (c[:, 2] > bed_z + 0.6)
    return a[down].sum() / a.sum()


def islands(m):
    lo, hi = m.bounds
    counts = []
    for z in np.arange(lo[2] + 8, hi[2] - 8, 5):
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is not None:
            counts.append(len(sec.to_2D()[0].polygons_full))
    return np.median(counts), max(counts)


def free_heights(m, step=10.0, side_x=29.0):
    """At each print height z, how tall the left and right sides stand free: z minus the highest
    point where something already printed joins them. A join is a path from one side to the
    other through material below z that doesn't run down either side (|x| > side_x) below
    that point."""
    man = mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                              tri_verts=np.asarray(m.faces, np.uint32)))
    lo, hi = m.bounds
    big = 400.0
    cross_box = mf.Manifold.cube((2 * side_x, big, 2 * big)).translate((-side_x, -big / 2, -big))
    out = []
    a_prev = lo[2]
    for z in np.arange(lo[2] + step, hi[2] + 0.01, step):
        below = man ^ mf.Manifold.cube((big, big, z - lo[2] + 1)).translate(
            (-big / 2, -big / 2, lo[2] - 1))
        cross = below ^ cross_box

        def joined(a):
            sides = below ^ mf.Manifold.cube((big, big, z - a)).translate((-big / 2, -big / 2, a))
            sides = sides - cross_box
            for part in (cross + sides).decompose():
                bb = part.bounding_box()
                if bb[0] < -side_x - 1 and bb[3] > side_x + 1:
                    return True
            return False
        a = z - 2.0
        while a > a_prev and not joined(a):
            a -= 4.0
        a = max(a, lo[2])
        a_prev = a
        out.append((round(float(z - lo[2])), round(float(z - a))))
    return out


def mesh_report(name):
    L = Layout()
    back, front = load(name, "back"), load(name, "front")
    lo = np.minimum(back.bounds[0], front.bounds[0])
    hi = np.maximum(back.bounds[1], front.bounds[1])
    length, width, height = hi[2] - lo[2], hi[0] - lo[0], hi[1] - lo[1]
    print(f"== {name}")
    print(f"size: {length:.0f} long x {height:.0f} tall x {width:.0f} wide mm; "
          f"Shelf depth {SHELF_DEPTH:.0f} -> {SHELF_DEPTH - length:.0f} mm to spare")
    for which, m in (("back", back), ("front", front)):
        bed = m.bounds[0][2]
        med, mx = islands(m)
        fh = free_heights(m)
        worst = max(fh, key=lambda t: t[1])
        over = [z for z, h in fh if h > 45]
        print(f"{which}: {m.volume / 1000:.0f} cm3, overhang {100 * overhang_share(m, bed):.1f}%, "
              f"islands/layer {med:.0f} (max {mx}), sides stand free up to {worst[1]:.0f} mm "
              f"(at {worst[0]:.0f} mm print height); over 45 mm at: {over or 'none'}")


def seam_report(name):
    """The Seam faces: their area, and the thinnest wall round each Pin hole, 2 mm in."""
    import joint
    from shapely.geometry import Point
    L = Layout()
    zs = L.length / 2
    for which, z in (("back", zs - 2.0), ("front", zs + 2.0)):
        m = load(name, which)
        face = m.section(plane_origin=[0, 0, zs + (-0.05 if which == "back" else 0.05)],
                         plane_normal=[0, 0, 1])
        polys = face.to_2D()[0].polygons_full if face is not None else []
        area = sum(p.area for p in polys)
        contacts = len(polys)
        sec = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        p2, T = sec.to_2D()
        inv = np.linalg.inv(T)
        walls = []
        polys = p2.polygons_full
        for poly in polys:
            for interior in poly.interiors:
                c = np.array(interior.centroid.coords[0])
                r = np.sqrt(abs(trimesh.path.polygons.Polygon(interior).area) / np.pi)
                if abs(r - (joint.PIN_R + joint.CLEARANCE)) < 0.4:
                    walls.append(poly.exterior.distance(Point(c)) - r)
        print(f"Seam {which}: {contacts} contacts, {area:.0f} mm2, Pin holes {len(walls)}, thinnest wall "
              f"{min(walls):.1f} mm" if walls else f"Seam {which}: face {area:.0f} mm2, NO Pin holes found")


def graph_report(seed):
    import grown
    L = Layout()
    g = grown.build_graph(L, seed)
    eggs = grown.egg_samples(L)
    A, B, R, K = g.arrays()
    clash = {}
    steep_fail = {}
    for a, b, r, k in zip(A, B, R, K):
        if k not in ("rail", "guide") and grown.segment_hits_egg(a, b, r, eggs, skip_ends=
                                                               6.0 if k in ("support", "tie") else 0):
            clash[k] = clash.get(k, 0) + 1
        on_bed = max(a[2], b[2]) < grown.BACK_Z + 1
        ang = grown.angle_from_z(a, b)
        bridge = ang >= 80 and np.linalg.norm(b - a) <= grown.MAX_BRIDGE
        if k not in ("rail", "rim", "frame") and not on_bed and not grown.ok_angle(a, b):
            steep_fail[k] = steep_fail.get(k, 0) + 1
    print(f"capacity: {len(L.egg_centres())} eggs")
    print(f"egg clashes (not Rails/guides): {clash or 'none'}")
    print(f"Struts breaking the print rule (45 deg; 56 if <= 24 mm; short bridges): {steep_fail or 'none'}")
    print(f"dangling ends: {len(grown.dangling(g))}")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        mesh_report(name)
        seam_report(name)
        if name.startswith("grown_"):
            graph_report(int(name.split("_")[1]))
