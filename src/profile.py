"""Side profiles: the near side of a Dispenser seen square-on, as a silhouette, and measurements
of its Lattice, to compare a Mini with the Original.

  uv run python src/profile.py <tag> [<tag> ...]

Writes build/mini/profile_<tags>.png (the Original first, then each Mini, same scale) and
prints, per model:
- strut_w    Strut width seen from the side (median and 10th-90th percentile, mm)
- holes      enclosed holes per 10,000 mm2 of side
- hole_d     a hole's equivalent diameter (median, mm)
- aspect     a hole's long axis over its short (median)
- lean       a hole's long axis, degrees from the length (median of |angle|)
- round      a hole's area over its convex hull's (median; 1 is convex)
- nodes      skeleton junctions per 10,000 mm2
- cover      material share of the side's outline
- diag       share of Strut length running 25-65 degrees from the length
"""

import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "mini"
ORIGINAL = ROOT / "reference" / "original_bio_egg_dispenser.3mf"
PX = 0.4  # mm per pixel


def original():
    m = trimesh.load(ORIGINAL, force="mesh")
    m.vertices -= [128.0, 69.54, 0]
    return m


SIDE = 0.55  # the side is what lies past this share of the half-width (0.88: the skin alone)


def silhouette(m, px=PX):
    """The side x > SIDE of the half-width, seen along -x: image rows y (up), columns z."""
    half = m.bounds[1][0]
    tri = m.triangles
    keep = tri[:, :, 0].mean(axis=1) > SIDE * half
    tri = tri[keep]
    lo = m.bounds[0]
    W = int((m.bounds[1][2] - lo[2]) / px) + 3
    H = int((m.bounds[1][1] - lo[1]) / px) + 3
    img = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(img)
    for t in tri:
        zs = (t[:, 2] - lo[2]) / px + 1
        ys = H - 1 - ((t[:, 1] - lo[1]) / px + 1)
        d.polygon(list(zip(zs, ys)), fill=1)
    return np.array(img, bool)


def measure(sil, px=PX):
    mat = ndimage.binary_closing(sil, iterations=1)
    outline = ndimage.binary_fill_holes(mat)
    holes = outline & ~mat
    lab = label(holes)
    props = [p for p in regionprops(lab) if p.area * px * px > 12]
    area = outline.sum() * px * px
    dt = ndimage.distance_transform_edt(mat) * px
    sk = skeletonize(mat)
    widths = 2 * dt[sk]
    nb = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    junction = label(sk & (nb >= 3))
    n_nodes = junction.max()
    asp = [p.axis_major_length / max(p.axis_minor_length, 1) for p in props]
    lean = [abs(90 - abs(np.degrees(p.orientation))) for p in props]   # 0 = along the length
    # Strut direction: the skeleton's local direction (PCA over a 9-px window)
    ys, xs = np.nonzero(sk)
    ang = []
    pts = np.column_stack([xs, ys])
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    for k in range(0, len(pts), 3):
        nb_i = tree.query_ball_point(pts[k], 5)
        if len(nb_i) < 5:
            continue
        q = pts[nb_i] - pts[nb_i].mean(0)
        w, v = np.linalg.eigh(q.T @ q)
        dx, dy = v[:, 1]
        ang.append(abs(np.degrees(np.arctan2(abs(dy), abs(dx)))))
    ang = np.array(ang)
    return dict(
        diag=float(np.mean((ang > 25) & (ang < 65))) if len(ang) else 0,
        strut_w=(np.percentile(widths, 10), np.median(widths), np.percentile(widths, 90)),
        holes=len(props) / area * 1e4,
        hole_d=float(np.median([p.equivalent_diameter_area * px for p in props])) if props else 0,
        aspect=float(np.median(asp)) if asp else 0,
        lean=float(np.median(lean)) if lean else 0,
        round=float(np.median([p.solidity for p in props])) if props else 0,
        nodes=n_nodes / area * 1e4,
        cover=mat.sum() / outline.sum(),
    )


def fmt(name, s):
    w = s["strut_w"]
    return (f"{name:10s} strut_w {w[1]:4.1f} ({w[0]:.1f}-{w[2]:.1f})  holes {s['holes']:4.1f}  "
            f"hole_d {s['hole_d']:5.1f}  aspect {s['aspect']:4.2f}  lean {s['lean']:4.1f}  "
            f"round {s['round']:4.2f}  nodes {s['nodes']:4.1f}  cover {s['cover']:4.2f}  "
            f"diag {s['diag']:4.2f}")


def score(s, ref):
    """Relative distance to the Original over the measurements that set the look."""
    keys = [("hole_d", 1.0), ("aspect", 1.0), ("round", 2.0), ("nodes", 1.0), ("cover", 1.5),
            ("diag", 2.0)]
    d = sum(w * abs(s[k] - ref[k]) / max(abs(ref[k]), 1e-6) for k, w in keys)
    d += abs(s["strut_w"][1] - ref["strut_w"][1]) / ref["strut_w"][1]
    d += abs(s["lean"] - ref["lean"]) / 45.0
    return d


def main(tags):
    ref_sil = silhouette(original())
    ref = measure(ref_sil)
    print(fmt("original", ref))
    sils = [ref_sil]
    for t in tags:
        m = trimesh.load(OUT / f"{t}.stl")
        sil = silhouette(m)
        s = measure(sil)
        print(fmt(t, s), f"  score {score(s, ref):.2f}")
        sils.append(sil)
    gap = 30
    H = max(s.shape[0] for s in sils)
    W = sum(s.shape[1] for s in sils) + gap * (len(sils) - 1)
    canvas = np.ones((H, W), bool)
    x = 0
    for s in sils:
        canvas[H - s.shape[0]:, x:x + s.shape[1]] = ~s
        x += s.shape[1] + gap
    path = OUT / f"profile_{'_'.join(tags)}.png"
    Image.fromarray((canvas * 255).astype(np.uint8)).save(path)
    print(path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].startswith("side="):
        SIDE = float(args.pop(0)[5:])
    main(args)
