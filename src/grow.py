"""Grow: the Lattice as rounded holes in a skin, rather than Struts on a net.

The cage's outside is the Tube (grown.py), its sides set at the Side guides' plane. Unrolled, the
Tube is a flat chart (z along the Dispenser, u around it). On the chart:

1. Cells: seeds relaxed into an even Voronoi pattern (Lloyd), a little longer along z. The Side
   guides are veins: cells may not cross them.
2. Holes: each cell shrinks by a Strut radius, gets a printable roof (its ceiling leans no
   further than ROOF from z, a teardrop that fits inside the cell) and its corners rounded. The
   rounding is what webs the Nodes.
3. Material is whatever no hole covers: round Struts where two holes are close, flat webs where
   three meet.

The skin becomes a distance field: a point at chart distance e from the nearest hole and
height n off the Tube is inside if sqrt(max(r - e, 0)^2 + n^2) < r. It blends with the Rails,
Side guides and their supports (grown.py's Strut graph) and is meshed with marching cubes.
"""

import math
import time
from dataclasses import dataclass, field, replace

import manifold3d as mf
import numpy as np
import shapely
from scipy import ndimage
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union
from skimage.draw import polygon as fill_polygon
from skimage.measure import find_contours, marching_cubes

import grown
from grown import Graph, Tube, egg_samples, hits_egg, segment_hits_egg, angle_from_z
from layout import BOTTOM_Y, GUIDE_R, GUIDE_X, RAIL_R, RAIL_X, XL, Layout


@dataclass
class Look:
    cell: float = 36.0      # seed spacing: the Original's cells are 35-40 mm
    aniso: float = 1.2      # cells this much longer along z than around
    lloyd: int = 40         # relaxation steps
    r: float = 2.2          # Strut radius: half the material between two holes
    fillet: float = 6.0     # hole corner rounding: sets the webs at the Nodes
    gap: float = 0.0        # the skin stands this far outside the Side guides (0: they are in it)
    tie_step: float = 22.0  # ... and holds them with a tie this often
    fold: float = 0.0       # chevrons: side cell edges lean atan(1/fold) from the length
    fold_round: float = 8.0 # ... rounded this much where they meet at the waist
    method: str = "field"   # the skin: "field" (distance field) or "subd" (picture frames +
                            # Catmull-Clark, subd.py)
    frame_w: float = 2.6    # subd: a picture frame's width (half a Strut, before subdivision)
    frame_h: float = 2.4    # subd: half the skin's thickness
    levels: int = 3         # subd: Catmull-Clark steps
    cage_tol: float = 4.0   # subd: the cage's lines simplified to this (mm): coarser, more flare
    shear: float = 0.0      # side cells lean: edges run atan(1/shear) from the length, as chevrons
    lean_min: float = 0.0   # cell edges lean at least this far from the length (diagonal Struts)
    roof: float = 56.0      # a hole's ceiling leans at most this far from z (ADR 0005)
    web: float = 0.5        # a web's thickness at its middle, as a share of a Strut's
    open_scale: float = 1.6 # cells on the top and bottom (past the shoulder and shelf) this much bigger
    warp: float = 5.0       # the cells' edges bow this much (mm, rms): Struts curve
    warp_len: float = 50.0  # ... over this length
    smooth: float = 0.6     # blur of the chart's distance map (mm): Struts swell into Nodes
    field_blur: float = 0.7 # blur of the 3D field before meshing (voxels): a smooth surface
    plate: float = 6.0      # material further than this from every hole gets a hole punched
    pinch: float = 5.0      # the waist between the Levels, this far in
    blend: float = 1.5      # skin to Rails/guides smooth-union width
    veins: str = "shoulder,shelf"   # Tube edges the cells may not cross (plus the guides)
    seed: int = 1


BACK_Z = 2.0                # back rim centre; the bed cut flattens it
OPEN_ROUND = 10.0           # corner rounding of the loading and finger openings
SOFT = 1.0                  # how softly a Strut turns into a web (mm)
LOBE = 7.0                 # the sides stay at the guides' plane this far above and below them
RES = 0.25                  # chart pixel
VOXEL = 0.4


class Skin(Tube):
    """The Tube with its lobes on the Side guides: x = GUIDE_X at both Levels' egg-centre
    heights, pinched in between."""

    def __init__(self, L, look):
        super().__init__(L)
        self.look = look

    def half_width(self, z):
        return GUIDE_X + self.look.gap

    def outline(self, z):
        """As Tube.outline, cached every 0.1 mm of z (not 0.5): the rounded ends are steep."""
        key = round(self.clamp(z) * 10) / 10
        if key in self._cache:
            return self._cache[key]
        from scipy.interpolate import CubicSpline
        right, names = self.control(key)
        loop = np.array(right + [(-x, y) for x, y in right[-2:0:-1]] + [right[0]])
        t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))]
        spline = CubicSpline(t, loop, bc_type="periodic")
        tt = np.linspace(0, t[-1], 481)
        pts = spline(tt)
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        named = {n: np.interp(tk, tt, d) / d[-1] for n, tk in zip(names, t) if n}
        self._cache[key] = (pts, d, named)
        return self._cache[key]

    def control(self, z):
        L = self.L
        W, T, B = GUIDE_X + self.look.gap, self.top(z), self.bottom(z)
        zc = min(max(z, L.z_column), L.z_front)
        lc = float(L.lower_y(zc))
        uc = float(L.upper_y(zc))
        lc = min(max(lc, B + 14), T - 50)
        uc = min(max(uc, lc + 36), T - 16)
        waist = (lc + uc) / 2
        tz = min(max((z - 40.0) / 35.0, 0.0), 1.0)        # no waist where eggs drop at the Turn
        pinch = self.look.pinch * tz * tz * (3 - 2 * tz)
        f = LOBE                                           # flat over the egg ends
        pts = [(0.0, B), (0.55 * W, B), (W - 3.5, B + 5.0), (W, lc - f), (W, lc), (W, lc + f),
               (W - pinch, waist), (W, uc - f), (W, uc), (W, uc + f), (W - 7.0, T - 7.0),
               (0.5 * W, T - grown.DOME), (0.0, T)]
        names = (None, None, "shelf", None, "lower_lobe", None, "waist", None, "upper_lobe",
                 None, "shoulder", None, None)
        return pts, names


# --- The chart ---

class Chart:
    """The Tube unrolled: v along it, u = s * P0 around (P0 the perimeter at mid-length). v is
    length measured on the Tube from the back rim (v = z0 there), so where the Tube's ends
    round off, the chart doesn't squeeze them: v = z wherever the Tube runs straight."""

    def __init__(self, L, skin, z0, z1, P0, dz=0.1, nu=1080):
        self.L, self.skin, self.z0, self.z1, self.P0 = L, skin, z0, z1, P0
        self.zg = np.arange(z0 - 8, z1 + 8 + dz, dz)
        self.ug = np.linspace(0, P0, nu + 1)
        s = self.ug / P0
        S = np.zeros((len(self.zg), nu + 1, 2))
        for i, z in enumerate(self.zg):
            pts, d, _ = skin.outline(z)
            S[i, :, 0] = np.interp(s * d[-1], d, pts[:, 0])
            S[i, :, 1] = np.interp(s * d[-1], d, pts[:, 1])
        step = np.sqrt(np.sum(np.diff(S, axis=0) ** 2, axis=2) + dz * dz)
        V = np.vstack([np.zeros((1, nu + 1)), np.cumsum(step, axis=0)])
        i0 = int(round((z0 - self.zg[0]) / dz))
        self.V = V - V[i0] + z0                         # v at (zg, ug)
        self.vg = np.arange(self.V.min(), self.V.max(), dz)
        self.Zv = np.column_stack([np.interp(self.vg, self.V[:, j], self.zg)
                                   for j in range(nu + 1)])
        self.dz, self.du = dz, P0 / nu
        self.Sx, self.Sy = S[:, :, 0], S[:, :, 1]

    def _lookup(self, table, a0, da, a, u):
        a, u = np.broadcast_arrays(np.asarray(a, float), np.asarray(u, float))
        out = ndimage.map_coordinates(table, [(np.ravel(a) - a0) / da,
                                              (np.ravel(u) % self.P0) / self.du],
                                      order=1, mode="nearest")
        return out.reshape(a.shape) if a.ndim else float(out[0])

    def v_of(self, z, u):
        return self._lookup(self.V, self.zg[0], self.dz, z, u)

    def z_of(self, v, u):
        return self._lookup(self.Zv, self.vg[0], self.dz, v, u)

    def v_front(self, u):
        return self.v_of(np.full(np.shape(u), self.z1), u)

    def point(self, v, u):
        return self.skin.at(float(self.z_of(v, u)), (u / self.P0) % 1.0)

    def points(self, Vs, U):
        Vs, U = np.ravel(Vs), np.ravel(U)
        Z = self.z_of(Vs, U)
        return np.array([self.skin.at(z, (u / self.P0) % 1.0) for z, u in zip(Z, U)])

    def locate(self, p):
        """(v, u) of the Tube point nearest p, in p's cross-section."""
        pts, d, _ = self.skin.outline(p[2])
        k = int(np.argmin(np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])))
        u = d[k] / d[-1] * self.P0
        return float(self.v_of(p[2], u)), u

    def domain(self):
        """The chart between the back and front rims, three tiles round."""
        us = np.linspace(-self.P0, 2 * self.P0, 1081)
        front = self.v_front(us)
        return Polygon(list(zip(np.full(len(us), self.z0), us)) +
                       list(zip(front[::-1], us[::-1])))


def tiled(geom, P0):
    return unary_union([shapely.affinity.translate(geom, 0, k * P0) for k in (-1, 0, 1)])


def mask_polygons(chart, pred, step=1.0, grow=0.0):
    """Polygons (chart coords) round the chart points whose Tube point satisfies pred."""
    zs = np.arange(chart.z0 - 6, chart.vg[-1] - 2, step)
    us = np.arange(-step, chart.P0 + step, step)
    Z, U = np.meshgrid(zs, us, indexing="ij")
    P = chart.points(Z, U)
    m = pred(P).reshape(Z.shape)
    if not m.any():
        return Polygon()
    polys = []
    for c in find_contours(np.pad(m, 1).astype(float), 0.5):
        c = c - 1
        poly = Polygon(np.column_stack([zs[0] + c[:, 0] * step, us[0] + c[:, 1] * step]))
        if poly.is_valid and poly.area > step * step:
            polys.append(poly)
    out = unary_union(polys)
    return out.buffer(grow) if grow else out


def openings(chart, look, eggs):
    """Holes the Lattice keeps open, as chart polygons whose edges become Struts: loading at the
    top front (open to the front), fingers beside the front egg, and wherever the skin would
    touch an egg."""
    L, skin = chart.L, chart.skin
    W = GUIDE_X
    z_open = L.z_front - 70.0
    y_low_top = float(L.lower_y(L.z_front)) + XL["diameter"] / 2 + 3

    def loading(P):
        x, y, z = P.T
        tops = np.array([skin.top(zz) for zz in z])
        return (z > z_open) & (y > tops - 11) & (np.abs(x) < 0.72 * W)

    def fingers(P):
        x, y, z = P.T
        bots = np.array([skin.bottom(zz) for zz in z])
        return (z > L.z_front - 12) & (z < chart.z1) & (np.abs(x) > 20) & \
            (y > bots + 8) & (y < y_low_top)

    def egg(P):
        return hits_egg(P, look.r, eggs, margin=1.0)

    # the loading opening runs off the chart's front edge, so no rim bridges over it
    rnd = lambda p: p.buffer(-OPEN_ROUND).buffer(OPEN_ROUND)
    out = [rnd(mask_polygons(chart, loading)), rnd(mask_polygons(chart, fingers)),
           mask_polygons(chart, egg, grow=look.r + 0.5)]
    return unary_union([o for o in out if not o.is_empty])


def veins(chart, L, look):
    """Lines the cells may not cross, as chart polylines: the Side guides, and the Tube's named
    edges listed in look.veins (shoulder, shelf, waist)."""
    lines = []
    for s in (-1, 1) if look.gap == 0 else ():          # guides in the skin's plane
        pts = [chart.locate(p) for p in L.guide(s)]
        lines.append(shapely.LineString(pts))
    zs = np.linspace(chart.z0, chart.z1 + 6, 60)
    for name in [v for v in look.veins.split(",") if v]:
        for side in (-1, 1):
            u = []
            for z in zs:
                f = chart.skin.outline(z)[2][name]
                u.append((f if side > 0 else 1 - f) * chart.P0)
            v = chart.v_of(zs, np.array(u))
            lines.append(shapely.LineString(np.column_stack([v, u])))
    return lines


# --- Cells ---

def mirror_u(u, P0):
    """The Dispenser's mirror (x -> -x) on the chart: s -> 1 - s."""
    return (P0 - np.asarray(u)) % P0


def mirrored(half, P0, eps):
    """All seeds from the right half's (u in [0, P0/2]); seeds on the mirror lines (u = 0, P0/2)
    stand for themselves. Returns seeds and, for each, its half seed and whether mirrored."""
    full, src, flip = [], [], []
    for k, (z, u) in enumerate(half):
        full.append((z, u)), src.append(k), flip.append(False)
        if min(u, abs(u - P0 / 2)) > eps:
            full.append((z, mirror_u(u, P0))), src.append(k), flip.append(True)
    return np.array(full), np.array(src), np.array(flip)


def poisson_half(domain, spacing, P0, rng, tries=8000):
    """Dart-throwing seeds in the right half, (z/aniso, u), spaced from each other and from
    their mirror images by spacing(u) (the pair's mean)."""
    x0, _, x1, _ = domain.bounds
    half = []
    for _ in range(tries):
        p = np.array([rng.uniform(x0, x1), rng.uniform(0, P0 / 2)])
        if not domain.contains(Point(p)):
            continue
        sp = spacing(p[1])
        if min(p[1], abs(p[1] - P0 / 2)) < 0.3 * sp:
            p[1] = 0.0 if p[1] < P0 / 4 else P0 / 2
        full, _, _ = mirrored(half, P0, 1e-6) if half else (np.zeros((0, 2)), 0, 0)
        cand = np.array([p, [p[0], mirror_u(p[1], P0)]])
        if len(full):
            d = np.abs(full[None, :, :] - cand[:, None, :])
            d[..., 1] = np.minimum(d[..., 1], P0 - d[..., 1])
            need = 0.5 * (sp + np.array([spacing(u) for u in full[:, 1]]))[None, :]
            if np.any(np.hypot(d[..., 0], d[..., 1]) < need):
                continue
        if p[1] not in (0.0, P0 / 2) and abs(p[1] - mirror_u(p[1], P0)) < sp:
            continue
        half.append(p)
    return np.array(half)


def voronoi(seeds, P0):
    tiles = np.concatenate([seeds + [0, k * P0] for k in (-1, 0, 1)])
    far = np.array([[-1e4, -1e4], [-1e4, 1e4], [1e4, -1e4], [1e4, 1e4]])
    return Voronoi(np.concatenate([tiles, far]))


def cells_of(vor, n):
    """The Voronoi cell of each of the n middle-tile seeds."""
    return [Polygon(vor.vertices[vor.regions[vor.point_region[i]]]).convex_hull
            for i in range(n, 2 * n)]


def ceiling_push(vor, n, a, theta, lean_min=0.0):
    """Pushes that tilt edges toward the diagonal: where the edge between two cells leans
    further than theta from z (real space), move the two seeds apart around the Tube; where it
    leans less than lean_min (runs along the length), move them apart along it."""
    push = np.zeros((n, 2))
    for (p, q), (v0, v1) in zip(vor.ridge_points, vor.ridge_vertices):
        if v0 < 0 or v1 < 0 or p >= 3 * n or q >= 3 * n:
            continue
        i, j = p % n, q % n
        if (p // n != 1) and (q // n != 1):
            continue
        e = vor.vertices[v1] - vor.vertices[v0]
        dz, du = e[0] * a, e[1]
        length = math.hypot(dz, du)
        ang = math.atan2(abs(du), abs(dz))
        if length < 4.0:
            continue
        sp, sq = vor.points[p], vor.points[q]
        if ang > theta:
            side = np.sign(sp[1] - sq[1]) or 1.0
            d = 0.35 * (ang - theta) * math.hypot(*(sp - sq))
            push[i, 1] += side * d / 2
            push[j, 1] -= side * d / 2
        elif ang < lean_min:
            side = np.sign(sp[0] - sq[0]) or (1.0 if i < j else -1.0)
            d = 0.35 * (lean_min - ang) * math.hypot(*(sp - sq)) / a
            push[i, 0] += side * d / 2
            push[j, 0] -= side * d / 2
    return push


def relax(domain, look, P0, rng, bands=None):
    """Lloyd, mirror-symmetric: seeds to the centroids of their cells (the piece on the seed's
    side of the veins and openings), plus a sideways push that tilts flat cell tops into
    printable roofs."""
    a = look.aniso
    eps = 0.3 * look.cell
    dom = shapely.affinity.scale(domain, 1 / a, 1, origin=(0, 0))
    def spacing(u):
        """Bigger cells on the top and bottom: past the shelf and shoulder lines."""
        if bands is None:
            return look.cell
        f = min(u % P0, P0 - u % P0) / P0
        return look.cell * (look.open_scale if f < bands[0] or f > bands[1] else 1.0)
    half = poisson_half(dom, spacing, P0, rng)
    theta = math.radians(look.roof)
    for it in range(look.lloyd):
        full, src, flip = mirrored(half, P0, 1e-6)
        n = len(full)
        vor = voronoi(full, P0)
        cells = cells_of(vor, n)
        push = ceiling_push(vor, n, a, theta, math.radians(look.lean_min)) \
            if it >= look.lloyd // 4 and look.fold == 0 else np.zeros((n, 2))
        new = half.copy()
        acc = np.zeros_like(half)
        cnt = np.zeros(len(half))
        for i, c in enumerate(cells):
            k = src[i]
            piece = c.intersection(dom)
            if piece.is_empty:
                continue
            if isinstance(piece, MultiPolygon):
                sp = Point(full[i])
                piece = min(piece.geoms, key=lambda g: g.distance(sp))
            target = np.array([piece.centroid.x, piece.centroid.y]) + push[i]
            if flip[i]:
                target[1] = mirror_u(target[1], P0)
                if target[1] > P0 * 0.75:
                    target[1] -= P0
            acc[k] += target
            cnt[k] += 1
        ok = cnt > 0
        new[ok] = acc[ok] / cnt[ok, None]
        on_line = np.array([min(abs(u), abs(u - P0 / 2)) < 1e-6 for u in half[:, 1]])
        new[on_line, 1] = half[on_line, 1]
        new[:, 1] = np.clip(new[:, 1], 0, P0 / 2)
        snap = ~on_line & (np.minimum(new[:, 1], P0 / 2 - new[:, 1]) < 0.5 * eps)
        new[snap, 1] = np.where(new[snap, 1] < P0 / 4, 0.0, P0 / 2)
        moved = np.max(np.abs(new - half))
        half = new
        if moved < 0.05 and it > look.lloyd // 4:
            break
    full, _, _ = mirrored(half, P0, 1e-6)
    cells = cells_of(voronoi(full, P0), len(full))
    return [shapely.affinity.scale(c, a, 1, origin=(0, 0)) for c in cells]


def named_line(chart, name, side, n=80):
    """A Tube edge (shelf, waist, shoulder) on the chart, (v, u), past both rims."""
    zs = np.linspace(chart.z0 - 8, chart.z1 + 8, n)
    u = np.array([chart.skin.outline(z)[2][name] for z in zs])
    u = (u if side > 0 else 1 - u) * chart.P0
    return np.column_stack([chart.v_of(zs, u), u])


def bands(chart, look):
    """The Tube cut lengthwise into bands at the shelf, waist and shoulder lines, each with its
    shear: the sides' Struts lean forward toward the waist from above and below (chevrons
    pointing to the front, as the Original's); the top and bottom don't lean.
    Returns [(polygon over three tiles, shear)]."""
    P0, K = chart.P0, look.shear
    L = {(n, s): named_line(chart, n, s) for n in ("shelf", "waist", "shoulder") for s in (-1, 1)}
    shift = lambda xy, du: xy + [0, du]

    def between(lo, hi):
        return Polygon(np.vstack([lo, hi[::-1]])).buffer(0)
    parts = [
        (between(shift(L["shelf", -1], -P0), L["shelf", 1]), 0.0),       # bottom
        (between(L["shelf", 1], L["waist", 1]), K),                       # right, lower
        (between(L["waist", 1], L["shoulder", 1]), -K),                   # right, upper
        (between(L["shoulder", 1], L["shoulder", -1]), 0.0),              # top
        (between(L["shoulder", -1], L["waist", -1]), K),                  # left, upper
        (between(L["waist", -1], L["shelf", -1]), -K),                    # left, lower
    ]
    return [(tiled(poly, P0), k) for poly, k in parts]


def relax_banded(domain, look, P0, rng, band_list, spacing):
    """Lloyd, mirror-symmetric, each band's cells drawn in its own sheared metric: a cell's
    neighbours along the band are separated by edges leaning atan(1/shear) from the length."""
    a = look.aniso
    eps = 0.3 * look.cell
    dom_s = shapely.affinity.scale(domain, 1 / a, 1, origin=(0, 0))
    half = poisson_half(dom_s, lambda u: spacing(u) , P0, rng)
    half[:, 0] *= a
    far = np.array([[-1e4, -1e4], [-1e4, 1e4], [1e4, -1e4], [1e4, 1e4]])
    from shapely import prepared

    preps = [prepared.prep(b) for b, _ in band_list]

    def band_of(p):
        for b, pr in enumerate(preps):
            if pr.contains(Point(p)):
                return b
        return int(np.argmin([g.distance(Point(p)) for g, _ in band_list]))

    def cells_for(full):
        n = len(full)
        bid = np.array([band_of(p) for p in full])
        tiles = np.concatenate([full + [0, k * P0] for k in (-1, 0, 1)])
        cells = [Polygon()] * n
        for b, (poly, k) in enumerate(band_list):
            idx = np.nonzero(bid == b)[0]
            if not len(idx):
                continue
            tp = np.column_stack([(tiles[:, 0] - k * tiles[:, 1]) / a, tiles[:, 1]])
            vor = Voronoi(np.concatenate([tp, far]))
            for i in idx:
                reg = vor.regions[vor.point_region[n + i]]
                q = vor.vertices[reg]
                real = np.column_stack([q[:, 0] * a + k * q[:, 1], q[:, 1]])
                cells[i] = Polygon(real).convex_hull.intersection(poly)
        return cells

    for it in range(look.lloyd):
        full, src, flip = mirrored(half, P0, 1e-6)
        cells = cells_for(full)
        acc = np.zeros_like(half)
        cnt = np.zeros(len(half))
        for i, c in enumerate(cells):
            piece = c.intersection(domain)
            if piece.is_empty:
                continue
            if not isinstance(piece, Polygon):
                sp = Point(full[i])
                piece = min(getattr(piece, "geoms", [piece]), key=lambda g: g.distance(sp))
            target = np.array([piece.centroid.x, piece.centroid.y])
            if flip[i]:
                target[1] = mirror_u(target[1], P0)
                if target[1] > P0 * 0.75:
                    target[1] -= P0
            acc[src[i]] += target
            cnt[src[i]] += 1
        new = half.copy()
        ok = cnt > 0
        new[ok] = acc[ok] / cnt[ok, None]
        on_line = np.array([min(abs(u), abs(u - P0 / 2)) < 1e-6 for u in half[:, 1]])
        new[on_line, 1] = half[on_line, 1]
        new[:, 1] = np.clip(new[:, 1], 0, P0 / 2)
        snap = ~on_line & (np.minimum(new[:, 1], P0 / 2 - new[:, 1]) < 0.5 * eps)
        new[snap, 1] = np.where(new[snap, 1] < P0 / 4, 0.0, P0 / 2)
        moved = np.max(np.abs(new - half))
        half = new
        if moved < 0.05:
            break
    full, _, _ = mirrored(half, P0, 1e-6)
    return [c for c in cells_for(full) if not c.is_empty]


def fold_map(chart, look, nm):
    """The chevron fold: (v, u) -> (v - h(u), u), h = -fold * |u - waist| on the sides (rounded
    at the waist, flat past the shelf and shoulder, mirrored on the left). Straight cell edges in
    the folded chart become chevrons on the Tube, pointing to the front. Returns (to, back)."""
    P0 = chart.P0
    uw, us, uh = nm["waist"] * P0, nm["shelf"] * P0, nm["shoulder"] * P0
    c = look.fold_round

    def h(u):
        x = np.mod(u, P0)
        x = np.minimum(x, P0 - x)
        x = np.clip(x, us, uh)
        return -look.fold * np.sqrt((x - uw) ** 2 + c * c)

    def to(xy):
        return np.column_stack([xy[:, 0] - h(xy[:, 1]), xy[:, 1]])

    def back(xy):
        return np.column_stack([xy[:, 0] + h(xy[:, 1]), xy[:, 1]])

    def apply(fn, geom):
        return shapely.transform(shapely.segmentize(geom, 2.0), fn).buffer(0)
    return (lambda g: apply(to, g)), (lambda g: apply(back, g))


def warp_field(chart, look, rng, step=1.0):
    """A smooth displacement (dv, du) over the chart, periodic round the Tube and mirror-
    symmetric (dv even, du odd under u -> P0 - u)."""
    vs = np.arange(chart.z0 - 20, chart.vg[-1] + 20, step)
    n_u = int(round(chart.P0 / step))
    sig = look.warp_len / 4 / step
    D = []
    for _ in range(2):
        d = ndimage.gaussian_filter(rng.standard_normal((len(vs), n_u)), sig, mode=("nearest", "wrap"))
        D.append(d)
    m = (-np.arange(n_u)) % n_u                       # mirror column
    dv = (D[0] + D[0][:, m]) / 2
    du = (D[1] - D[1][:, m]) / 2
    scale = look.warp / max(np.sqrt(np.mean(dv ** 2 + du ** 2)), 1e-9)
    return vs, chart.P0 / n_u, dv * scale, du * scale


def warped(poly, field):
    """poly with every point moved by the warp field (shared edges move together)."""
    vs, du_step, dv, du = field
    poly = shapely.segmentize(poly, 2.0)
    xy = np.asarray(poly.exterior.coords)
    ci = [(xy[:, 0] - vs[0]) / (vs[1] - vs[0]), (xy[:, 1] % (du_step * dv.shape[1])) / du_step]
    mv = ndimage.map_coordinates(dv, ci, order=1, mode="grid-wrap")
    mu = ndimage.map_coordinates(du, ci, order=1, mode="grid-wrap")
    out = Polygon(np.column_stack([xy[:, 0] + mv, xy[:, 1] + mu]))
    return out if out.is_valid else out.buffer(0)


# --- Holes ---

def apex(poly, theta):
    """Where a hole's two roof lines, theta from z, meet over it (chart z is print up), and
    the lower of the z's where they touch it."""
    c, s = math.cos(theta), math.sin(theta)
    xy = np.asarray(poly.exterior.coords)          # (z, u)
    up, side = xy[:, 0], xy[:, 1]
    k1 = np.argmax(-c * side + s * up)
    k2 = np.argmax(c * side + s * up)
    h1 = -c * side[k1] + s * up[k1]
    h2 = c * side[k2] + s * up[k2]
    return (h1 + h2) / (2 * s), (h2 - h1) / (2 * c), min(up[k1], up[k2])


def roofed(poly, theta):
    """poly with a roof: the hull of its top and the apex (dents lower down stay)."""
    zt, ut, z_touch = apex(poly, theta)
    hull = shapely.convex_hull(shapely.union(poly, Point(zt, ut)))
    return poly.union(hull.intersection(box(z_touch, -1e5, zt + 1, 1e5)))


def printable(piece, theta, tol=0.05):
    """The largest teardrop (piece's lower part plus a roof theta from z) inside piece."""
    t = roofed(piece, theta)
    if t.difference(piece).area < tol * max(1.0, piece.length):
        return t
    z_lo, _, z_hi, _ = piece.bounds
    lo, hi = z_lo, z_hi
    best = Polygon()
    for _ in range(14):
        mid = (lo + hi) / 2
        cut = piece.intersection(box(z_lo - 1, -1e5, mid, 1e5))
        if cut.is_empty:
            lo = mid
            continue
        if isinstance(cut, MultiPolygon):
            cut = max(cut.geoms, key=lambda g: g.area)
        t = roofed(cut, theta)
        if t.difference(piece).area < tol * max(1.0, cut.length):
            best, lo = t, mid
        else:
            hi = mid
    return best


def holes(cells, domain, look):
    inner = domain.buffer(-look.r)
    out, cut = [], 0
    theta = math.radians(look.roof)
    for c in cells:
        allowed = c.buffer(-look.r).intersection(inner)
        for piece in getattr(allowed, "geoms", [allowed]):
            if piece.is_empty or piece.area < 4 or not isinstance(piece, Polygon):
                continue
            t = printable(piece, theta)
            if t.area < 0.97 * roofed(piece, theta).area:
                cut += 1
            h = t.buffer(-look.fillet).buffer(look.fillet)
            if not h.is_empty:
                out.append(h)
    return out, cut


def plate_holes(e, ezs, eus, chart, look):
    """Holes punched in plates: wherever material lies further than look.plate from every hole,
    a hole whose edge keeps a Strut's width (2r) from the others, roofed and rounded."""
    out = []
    m = e > 2 * look.r
    lab, n = ndimage.label(m)
    for k in range(1, n + 1):
        comp = lab == k
        if e[comp].max() < look.plate:
            continue
        for c in find_contours(np.pad(comp, 1).astype(float), 0.5):
            c = c - 1
            poly = Polygon(np.column_stack([ezs[0] + c[:, 0] * RES, eus[0] + c[:, 1] * RES]))
            if not poly.is_valid:
                poly = poly.buffer(0)
            for piece in getattr(poly, "geoms", [poly]):
                if piece.area < 10:
                    continue
                t = printable(piece.simplify(0.3), math.radians(look.roof))
                h = t.buffer(-look.fillet / 2).buffer(look.fillet / 2)
                if h.area > 6:
                    out.append(h)
    return out


def chart_field(chart, hole_polys, look, margin=12.0):
    """Signed distance to the nearest hole on the chart grid: positive in material."""
    zs = np.arange(chart.z0 - margin, chart.vg[-1], RES)
    us = np.arange(-margin, chart.P0 + margin, RES)
    mask = np.zeros((len(zs), len(us)), bool)
    mask[zs < chart.z0 - look.r] = True
    mask |= zs[:, None] > chart.v_front(us)[None, :] + look.r
    for h in hole_polys:
        for g in getattr(h, "geoms", [h]):
            for k in (-1, 0, 1):
                xy = np.asarray(g.exterior.coords)
                rr, cc = fill_polygon((xy[:, 0] - zs[0]) / RES, (xy[:, 1] + k * chart.P0 - us[0]) / RES,
                                      mask.shape)
                mask[rr, cc] = True
    e = ndimage.distance_transform_edt(~mask, sampling=RES) - \
        ndimage.distance_transform_edt(mask, sampling=RES)
    e = ndimage.gaussian_filter(e, look.smooth / RES)      # no pixel ripple; rounder forks
    return e, zs, us


# --- Skin field in 3D ---

def surface_samples(chart, step=0.3):
    """The Tube sampled evenly in the chart (v, u), with the chart's tangents and the normal."""
    vs = np.arange(chart.z0 - 5, chart.vg[-1] - 3, step)
    n_u = int(math.ceil(chart.P0 / step))
    us = np.arange(n_u) * chart.P0 / n_u
    Vv, Uu = np.meshgrid(vs, us, indexing="ij")
    Z = chart.z_of(Vv, Uu)
    ci = [(Z - chart.zg[0]) / chart.dz, Uu / chart.du]
    S = np.stack([ndimage.map_coordinates(chart.Sx, ci, order=1, mode="nearest"),
                  ndimage.map_coordinates(chart.Sy, ci, order=1, mode="nearest"), Z], axis=2)
    du = chart.P0 / n_u
    Su = (np.roll(S, -1, axis=1) - np.roll(S, 1, axis=1)) / (2 * du)
    Sv = np.gradient(S, step, axis=0)
    N = np.cross(Su, Sv)
    N /= np.linalg.norm(N, axis=2, keepdims=True)
    # outward: away from the cross-section's middle
    mid = S[:, :, :2].mean(axis=1, keepdims=True)
    flip = np.einsum("ijk,ijk->ij", N[:, :, :2], S[:, :, :2] - mid) < 0
    N[flip] *= -1
    return S, Sv, Su, N, vs, us


def skin_field(chart, e, ezs, eus, look, lo, shape, h=VOXEL):
    """f on the voxel grid (lo, shape): the skin's Struts and webs; 10 far from the skin."""
    S, Sz, Su, N, szs, sus = surface_samples(chart)
    e_spline = ndimage.spline_filter(e, order=3)
    flatS = S.reshape(-1, 3)
    # band: voxels near any sample
    idx = np.floor((flatS - lo) / h).astype(int)
    ok = np.all((idx >= 0) & (idx < shape), axis=1)
    hit = np.zeros(shape, bool)
    hit[tuple(idx[ok].T)] = True
    near = ndimage.distance_transform_edt(~hit, sampling=h) < look.r + 3
    vox = np.argwhere(near)
    P = lo + vox * h
    tree = cKDTree(flatS)
    f = np.full(shape, 10.0, np.float32)
    for c0 in range(0, len(P), 1_000_000):
        p = P[c0:c0 + 1_000_000]
        _, k = tree.query(p, workers=-1)
        i, j = np.unravel_index(k, S.shape[:2])
        M = np.stack([Sz[i, j], Su[i, j], N[i, j]], axis=2)       # columns
        sol = np.linalg.solve(M, (p - S[i, j])[:, :, None])[:, :, 0]
        uc = (sus[j] + sol[:, 1]) % chart.P0
        zc = szs[i] + sol[:, 0]                      # v
        n = sol[:, 2]
        ec = ndimage.map_coordinates(e_spline, [(zc - ezs[0]) / RES, (uc - eus[0]) / RES],
                                     order=3, mode="nearest", prefilter=False)
        t = softplus(look.r - ec, SOFT)
        # webs thin toward their middles, like a soap film between the Struts
        hw = look.r * (look.web + (1 - look.web) * np.exp(-softplus(ec - look.r, SOFT) / 2.0))
        f[tuple(vox[c0:c0 + 1_000_000].T)] = np.sqrt(t * t + n * n) - hw
    return f


def softplus(x, c):
    """A smooth max(x, 0): within about c of zero it rounds the corner."""
    return 0.5 * (x + np.sqrt(x * x + c * c))


def mesh_health(m):
    """(longest edge mm, edges over 3 mm, watertight)."""
    e = m.edges_unique_length
    return round(float(e.max()), 1), int((e > 3).sum()), bool(m.is_watertight)


def smin(a, b, k):
    hh = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * hh - k * hh * (1 - hh)


# --- Rails, Side guides, supports ---

def inner_graph(L, chart, e, ezs, eus, look, eggs, skin_anchors=None):
    g = Graph()
    for s in (-1, 1):
        g.polyline(L.lower_rail(s), RAIL_R, "rail")
        g.polyline(L.upper_rail(s), RAIL_R, "rail")
        g.polyline(L.guide(s), GUIDE_R, "guide")
    # Exit cradle loop under the front egg, as the Original's.
    zf = chart.z1
    zc0 = L.z_front - 30
    loop = [(18 * math.sin(a), BOTTOM_Y, zc0 + (zf + 3 - zc0) * (1 - math.cos(a)) / 2)
            for a in np.linspace(-math.pi, math.pi, 40)]
    g.polyline(loop, grown.FRAME_R, "frame")

    # Anchors: the skin's Strut centres.
    if skin_anchors is not None:
        anchors = skin_anchors
    else:
        sel = e > 0.6 * look.r
        sel[:, (eus < 0) | (eus >= chart.P0)] = False
        zi, ui = np.nonzero(sel)
        keep = ((ezs[zi] / 2.5).astype(int) * 1000 + (eus[ui] / 2.5).astype(int))
        _, first = np.unique(keep, return_index=True)
        anchors = chart.points(ezs[zi[first]], eus[ui[first]])
    anchors = np.concatenate([anchors] + [grown.resample(np.array(L.guide(s)), 10.0)
                                          for s in (-1, 1)])

    def support(a, prefer=None, max_len=60):
        a = np.asarray(a, float)
        d = np.linalg.norm(anchors - a, axis=1)
        for k in np.argsort(d):
            c = anchors[k]
            if d[k] < 8:
                continue
            if d[k] > max_len:
                break
            if angle_from_z(a, c) > grown.SUPPORT_ANGLE:
                continue
            if prefer is not None and np.sign(c[2] - a[2]) != prefer:
                continue
            if segment_hits_egg(a, c, grown.R_END, eggs, skip_ends=6.0):
                continue
            g.add(a, c, grown.R_MID, "support", rb=look.r * 0.9)
            return True
        return False

    missing = 0
    for s in (-1, 1):
        for i, z in enumerate(np.arange(L.upper_rail_back_z, L.z_front + 1, 34.0)):
            a = (s * RAIL_X, float(L.upper_rail_y(z)), z)
            missing += not (support(a, prefer=1 if i % 2 else -1) or support(a))
        missing += not support(L.upper_rail(s)[-1])
        for z in np.arange(L.z_front, 40, -34.0):
            a = (s * RAIL_X, float(L.lower_rail_y(z)), z)
            missing += not support(a)
        missing += not support(L.lower_rail(s)[0])

    # Side guides inside the skin (look.gap > 0): short ties out to the skin's Struts.
    if look.gap > 0:
        for s in (-1, 1):
            gp = grown.resample(np.array(L.guide(s)), 2.0)
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(gp, axis=0), axis=1))]
            for t in np.arange(4.0, d[-1] - 2, look.tie_step):
                p = gp[np.searchsorted(d, t)]
                side = anchors[anchors[:, 0] * s > GUIDE_X]
                dd = np.linalg.norm(side - p, axis=1)
                for k in np.argsort(dd)[:20]:
                    c = side[k]
                    if dd[k] > look.gap + 10:
                        break
                    if angle_from_z(p, c) <= 56 or dd[k] <= grown.MAX_BRIDGE:
                        g.add(p, c, 0.8 * GUIDE_R, "tie", rb=0.8 * look.r)
                        break
                else:
                    missing += 1

    # Back face rungs, on the bed, from the back wall Rails out to the skin.
    back = chart.skin.outline(BACK_Z)[0]
    wall = L.lower_rail(1)
    y0 = wall[wall[:, 2] < BACK_Z + 0.5][:, 1].min()
    for y in np.arange(y0 + 8, chart.skin.top(BACK_Z) - 6, 22.0):
        for sx in (-1, 1):
            side = back[back[:, 0] * sx > 0]
            xw = float(np.abs(side[np.argmin(np.abs(side[:, 1] - y)), 0]))
            a, b = (sx * RAIL_X, y, BACK_Z), (sx * xw, y, BACK_Z)
            if not segment_hits_egg(a, b, grown.FRAME_R, eggs, skip_ends=3.0):
                g.add(a, b, grown.FRAME_R, "back")
    return g, missing


# --- Build ---

def build_subd(L, chart, cells, vs, domain, look, eggs_s, t0):
    """The skin as a subdivision surface (subd.py), joined with the Rails, Side guides and
    supports (meshed from their distance field)."""
    import subd
    import trimesh
    V, T, anchors, n_faces = subd.skin_mesh(chart, cells, vs, domain, look.frame_w,
                                            look.frame_h, look.levels, look.cage_tol)
    skin_m = trimesh.Trimesh(V, T, process=True)
    t_chart = time.time() - t0
    g, missing = inner_graph(L, chart, None, None, None, look, eggs_s, skin_anchors=anchors)
    h = VOXEL
    A = np.array(g.seg_a + g.seg_b)
    lo = A.min(axis=0) - 6
    hi = A.max(axis=0) + 6
    shape = tuple(np.ceil((hi - lo) / h).astype(int) + 1)
    f_in, axes = grown.capsule_field(g, lo, shape, h)
    f_in[[0, -1]] = f_in[:, [0, -1]] = f_in[:, :, [0, -1]] = 10.0
    verts, faces, _, _ = marching_cubes(f_in, 0.0, spacing=(h, h, h))
    inner = trimesh.Trimesh(verts + lo, faces, process=True)
    import fast_simplification
    v2, f2 = fast_simplification.simplify(inner.vertices, inner.faces, target_reduction=0.6)
    inner = trimesh.Trimesh(v2, f2, process=True)

    def man(m):
        if m.volume < 0:
            m.invert()
        return mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                                   tri_verts=np.asarray(m.faces, np.uint32)))
    sk = man(skin_m)
    inn = man(inner)
    print("  manifold:", sk.status(), inn.status(), "skin watertight", skin_m.is_watertight)
    body = sk + inn
    big = 1000.0
    body = body ^ mf.Manifold.cube((big, big, big)).translate((-big / 2, 0, 0))
    out = body.to_mesh()
    mesh = trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts, process=True)
    print("  mesh:", mesh_health(mesh))
    from check import overhang_share
    hit = hits_egg(skin_m.vertices, 0.0, eggs_s, margin=0.0)
    stats = dict(faces=n_faces, supports_missing=missing, egg_clash_vertices=int(hit.sum()),
                 overhang=round(100 * overhang_share(mesh, 0.0), 1),
                 volume_cm3=round(mesh.volume / 1000, 1), t_chart=round(t_chart, 1),
                 t_total=round(time.time() - t0, 1))
    print(stats)
    return Result(mesh, cells, [], chart, stats)


@dataclass
class Result:
    mesh: object
    cells: list
    holes: list
    chart: Chart
    stats: dict = field(default_factory=dict)


def build(eggs=6, look=Look(), verbose=True):
    t0 = time.time()
    L = Layout(eggs=eggs)
    rng = np.random.default_rng(look.seed)
    skin = Skin(L, look)
    chart = Chart(L, skin, BACK_Z, L.length - 2.5, skin.perimeter(L.length / 2))
    print(f'  chart: front rim at v {chart.v_front(np.array([0, chart.P0 / 2])).round(1)}')
    eggs_s = egg_samples(L)

    opens = openings(chart, look, eggs_s)
    vs = veins(chart, L, look)
    rect = chart.domain()
    slits = tiled(unary_union([v.buffer(0.05) for v in vs]), chart.P0)
    domain = rect.difference(tiled(opens, chart.P0)).difference(slits)

    nm = skin.outline(L.length / 2)[2]
    if look.shear > 0:
        P0 = chart.P0

        def spacing(u):
            f = min(u % P0, P0 - u % P0) / P0
            open_band = f < nm["shelf"] or f > nm["shoulder"]
            return look.cell * (look.open_scale if open_band else 1.0)
        cells = relax_banded(domain, look, chart.P0, rng, bands(chart, look), spacing)
    elif look.fold > 0:
        to, back = fold_map(chart, look, nm)
        cells = relax(to(domain), look, chart.P0, rng, bands=(nm["shelf"], nm["shoulder"]))
        cells = [back(c) for c in cells]
    else:
        cells = relax(domain, look, chart.P0, rng, bands=(nm["shelf"], nm["shoulder"]))
    if look.warp > 0:
        wf = warp_field(chart, look, rng)
        cells = [warped(c, wf) for c in cells]
    if look.method == "subd":
        return build_subd(L, chart, cells, vs, rect.difference(tiled(opens, chart.P0)),
                          look, eggs_s, t0)
    hole_polys, cut = holes(cells, domain, look)
    open_holes = tiled(opens, chart.P0).buffer(-look.r)
    open_holes = open_holes.buffer(-look.fillet / 2).buffer(look.fillet / 2)
    all_holes = hole_polys + [open_holes]
    e, ezs, eus = chart_field(chart, all_holes, look)
    punched = 0
    for _ in range(2):
        extra = plate_holes(e, ezs, eus, chart, look)
        if not extra:
            break
        punched += len(extra)
        all_holes += extra
        e, ezs, eus = chart_field(chart, all_holes, look)
    t_chart = time.time() - t0

    g, missing = inner_graph(L, chart, e, ezs, eus, look, eggs_s)

    h = VOXEL
    lo = np.array([-GUIDE_X - 10, -1.0, -1.0])
    hi = np.array([GUIDE_X + 10, L.height + grown.ARCH + 8, L.length + 3])
    shape = tuple(np.ceil((hi - lo) / h).astype(int) + 1)
    f_skin = skin_field(chart, e, ezs, eus, look, lo, shape, h)
    f_in, axes = grown.capsule_field(g, lo, shape, h)
    f = smin(f_skin, f_in, look.blend)
    X, Y, Z = np.meshgrid(*axes, indexing="ij", sparse=True)
    f = np.maximum(f, -Y)
    f = np.maximum(f, -Z)
    f[[0, -1]] = f[:, [0, -1]] = f[:, :, [0, -1]] = 10.0
    if look.field_blur:
        f = ndimage.gaussian_filter(f, look.field_blur)      # smooth surface, no stray triangles
    verts, faces, _, _ = marching_cubes(f, 0.0, spacing=(h, h, h))
    import trimesh
    import fast_simplification
    raw = trimesh.Trimesh(verts + lo, faces, process=True)
    diag = {"raw": mesh_health(raw)}
    v2, f2 = fast_simplification.simplify(raw.vertices, raw.faces, target_reduction=0.7)
    mesh = trimesh.Trimesh(v2, f2, process=True)
    diag["simplified"] = mesh_health(mesh)
    print("  mesh:", diag)
    if mesh.volume < 0:
        mesh.invert()

    # egg clearance: skin vertices inside a swept egg, away from Rails and Side guides
    A, B, R, K = g.arrays()
    lines = np.concatenate([grown.resample(np.array(L.guide(s)), 2.0) for s in (-1, 1)] +
                           [L.lower_rail(s) for s in (-1, 1)] + [L.upper_rail(s) for s in (-1, 1)])
    hit = hits_egg(mesh.vertices, 0.0, eggs_s, margin=0.0)
    if hit.any():
        dl, _ = cKDTree(lines).query(mesh.vertices[hit])
        clash = int((dl > 4.0).sum())
    else:
        clash = 0
    from check import overhang_share
    stats = dict(cells=len(cells), holes=len(hole_polys), roof_cut=cut, punched=punched,
                 supports_missing=missing, egg_clash_vertices=clash,
                 overhang=round(100 * overhang_share(mesh, 0.0), 1),
                 volume_cm3=round(mesh.volume / 1000, 1),
                 t_chart=round(t_chart, 1), t_total=round(time.time() - t0, 1))
    if verbose:
        print(stats)
    return Result(mesh, cells, all_holes, chart, stats)
