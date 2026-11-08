"""Grown: the Dispenser generated from scratch around the shared egg path (layout.py), in the
Original's style.

The cage is a graph of Struts: the Rails and Side guides from layout, frame lines along the
cage's corners, and a Lattice that wraps the whole cage. The cage's outside is a tube: at each z
its cross-section is a rounded rectangle (rounded more at the top, as the Original), and its
top-back and top-front corners are rounded in side view. The Lattice is a Voronoi network
grown on that tube (cells a little longer along z than around), with curved, bone-shaped Struts
(thicker at their ends) blended into webbed Nodes.

Rules every Lattice Strut keeps:
- Printable: each half prints with z vertical. A Strut runs within 45 degrees of z, or within
  56 degrees if it is no longer than SHORT (the Original leans further, and sags), or it is a
  short bridge square to z between two Nodes that print. Every Node has a printable Strut
  reaching it from below.
- Clear of eggs: nothing but the Rails and Side guides enters the space an XL or Large egg
  sweeps along its path.

The graph becomes a signed distance field (log-sum-exp smooth union, ADR 0004), each half is cut
out and meshed, and the Seam joint is added.
"""

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import manifold3d as mf
import numpy as np
import trimesh
from scipy.spatial import Voronoi
from skimage.measure import marching_cubes

import joint
from layout import (BOTTOM_Y, GUIDE_R, LARGE, RAIL_R, RAIL_X, SHELL_X, XL, Layout)

BUILD = Path(__file__).resolve().parent.parent / "build"

# --- Look ---
SPACING = 38.0        # Voronoi seed spacing around the cage (the Original's cells: ~30-40 mm)
ANISOTROPY = 1.25     # cells this much longer along z than around
CURVE = 0.18          # a Strut's sideways bow, as a fraction of its length
R_MID, R_END = 1.9, 2.8   # Lattice Strut radius at its middle and at its Nodes (bone-shaped)
FRAME_R = 2.5         # corner frame lines, rims
SEAM_CLEAR = 6.0      # only the 4 Pin bosses, 4 Rails, 4 Side guides and 2 waist lines cross
                      # the Seam (14 contacts); everything else keeps this far from it
NODE_R = 0.0          # no Node spheres: the blend alone webs the joints
BLEND_K = 1.8         # smooth-union width: webs the joints
PINCH = 7.0           # the waist between the Levels, this far in from the lobes
DOME = 3.0            # the top's middle stands this much above its shoulders' line
BULGE = 4.0           # the sides swell this much at mid-length
ARCH = 6.0            # the top rises this much at mid-length
BACK_ROUND = 20.0     # side-view rounding of the top-back corner
BOTTOM_ROUND = 36.0   # ... the bottom-back corner, curving up as the Original's
FRONT_ROUND = 12.0    # ... the top-front corner

# --- Print rules ---
SHORT = 24.0          # Struts this short may lean up to SHORT_ANGLE from z
SHORT_ANGLE = 56.0
MAX_BRIDGE = 18.0     # a Strut square to z may bridge this far
EGG_MARGIN = 1.0
BACK_Z = 2.0          # back ring centre; the bed cut (z >= 0) flattens its foot
SUPPORT_ANGLE = 42.0  # Rail supports
TIE_SPACING = 38.0    # bottom V ties: one completes every this much print height, or less
VOXEL = 0.4


def ok_angle(a, b):
    """True if Strut a-b prints: steep, short and fairly steep, or a short bridge."""
    ang = angle_from_z(a, b)
    length = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
    return (ang <= 45.0 or (ang <= SHORT_ANGLE and length <= SHORT)
            or (ang >= 80.0 and length <= MAX_BRIDGE))


def angle_from_z(a, b):
    v = np.asarray(b, float) - np.asarray(a, float)
    return math.degrees(math.atan2(math.hypot(v[0], v[1]), abs(v[2])))


@dataclass
class Graph:
    """Struts as tapered segments (a, b, radius at a, radius at b). A chain's segments blend as
    one Strut (hard minimum); different chains and the Nodes blend smoothly."""
    seg_a: list = field(default_factory=list)
    seg_b: list = field(default_factory=list)
    seg_ra: list = field(default_factory=list)
    seg_rb: list = field(default_factory=list)
    seg_kind: list = field(default_factory=list)
    seg_chain: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    chains: int = 0

    def new_chain(self):
        self.chains += 1
        return self.chains - 1

    def add(self, a, b, r, kind, chain=None, rb=None):
        if chain is None:
            chain = self.new_chain()
        self.seg_a.append(np.asarray(a, float))
        self.seg_b.append(np.asarray(b, float))
        self.seg_ra.append(r)
        self.seg_rb.append(r if rb is None else rb)
        self.seg_kind.append(kind)
        self.seg_chain.append(chain)

    def polyline(self, pts, r, kind, step=6.0):
        pts = resample(np.asarray(pts, float), step)
        c = self.new_chain()
        for a, b in zip(pts[:-1], pts[1:]):
            self.add(a, b, r, kind, c)

    def bone(self, pts, kind):
        """A Lattice Strut along pts: radius R_END at both ends, R_MID in the middle."""
        pts = np.asarray(pts, float)
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        t = d / d[-1]
        r = R_MID + (R_END - R_MID) * (2 * t - 1) ** 2
        c = self.new_chain()
        for i in range(len(pts) - 1):
            self.add(pts[i], pts[i + 1], r[i], kind, c, r[i + 1])

    def arrays(self):
        """(a, b, radius, kind): radius is the larger end's, for checks."""
        return (np.array(self.seg_a), np.array(self.seg_b),
                np.maximum(self.seg_ra, self.seg_rb), np.array(self.seg_kind))


def resample(pts, step):
    """Split each segment of a polyline into pieces no longer than `step`, keeping its corners."""
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out)


# --- Egg clearance ---

def egg_samples(L):
    out = []
    for egg in (LARGE, XL):
        out.append((L.egg_path(egg, step=2.0), egg["length"] / 2, egg["diameter"] / 2))
    return out


def hits_egg(points, r, eggs, margin=EGG_MARGIN):
    hit = np.zeros(len(points), bool)
    for path, a, b in eggs:
        ax, bx = a + r + margin, b + r + margin
        dy = points[:, None, 1] - path[None, :, 0]
        dz = points[:, None, 2] - path[None, :, 1]
        q = (points[:, None, 0] / ax) ** 2 + (dy ** 2 + dz ** 2) / bx ** 2
        hit |= (q < 1).any(axis=1)
    return hit


def segment_hits_egg(a, b, r, eggs, skip_ends=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = np.linalg.norm(b - a)
    n = max(2, int(length / 1.5) + 1)
    t = np.linspace(0, 1, n)
    pts = a + t[:, None] * (b - a)
    keep = (t * length >= skip_ends) & ((1 - t) * length >= skip_ends)
    return bool(hits_egg(pts[keep], r, eggs).any()) if keep.any() else False


# --- The cage's tube ---

class Tube:
    """The cage's outer surface. (z, s): s runs around the cross-section from the bottom middle,
    counter-clockwise seen from the front, as a fraction of the way round.

    The cross-section is a smooth closed curve through a few points per side: the shelf edge,
    a lobe round each Level's eggs (widest at egg-centre height), a waist pinched in between
    them, a shoulder, and a domed top. Along z the sides bulge like a barrel, the top arches,
    and the back (top and bottom) and top-front corners round off."""

    NAMES = ("shelf", "lower_lobe", "waist", "upper_lobe", "shoulder")

    def __init__(self, L):
        self.L = L
        self._cache = {}

    def clamp(self, z):
        return min(max(float(z), 0.0), self.L.length - 2.5)

    def half_width(self, z):
        """At the lobes; never inside the Side guides (SHELL_X) plus a Strut's reach."""
        return SHELL_X + 2.0 + BULGE * math.sin(math.pi * self.clamp(z) / self.L.length)

    def top(self, z):
        L, z = self.L, self.clamp(z)
        t = float(L.top_y(z)) + ARCH * math.sin(math.pi * z / L.length)
        if z < BACK_ROUND:
            t -= BACK_ROUND - math.sqrt(max(0.0, BACK_ROUND ** 2 - (BACK_ROUND - z) ** 2))
        zf = L.length - 2.5
        if z > zf - FRONT_ROUND:
            dz = z - (zf - FRONT_ROUND)
            t -= FRONT_ROUND - math.sqrt(max(0.0, FRONT_ROUND ** 2 - dz ** 2))
        return t

    def bottom(self, z):
        z = self.clamp(z)
        b = BOTTOM_Y
        if z < BOTTOM_ROUND:
            b += BOTTOM_ROUND - math.sqrt(max(0.0, BOTTOM_ROUND ** 2 - (BOTTOM_ROUND - z) ** 2))
        return b

    def control(self, z):
        """The right half's control points, bottom middle to top middle, and their names."""
        L = self.L
        W, T, B = self.half_width(z), self.top(z), self.bottom(z)
        zc = min(max(z, L.z_column), L.z_front)
        lc = float(L.lower_y(zc)) + 1.0
        uc = float(L.upper_y(zc)) + 1.0
        lc = min(max(lc, B + 14), T - 50)
        uc = min(max(uc, lc + 36), T - 16)
        waist = (lc + uc) / 2
        # no waist where eggs drop through the Turn
        tz = min(max((z - 40.0) / 35.0, 0.0), 1.0)
        pinch = PINCH * tz * tz * (3 - 2 * tz)
        pts = [(0.0, B), (0.55 * W, B), (W - 3.5, B + 5.0), (W, lc), (W - pinch, waist),
               (W, uc), (W - 7.0, T - 7.0), (0.5 * W, T - DOME), (0.0, T)]
        names = (None, None, "shelf", "lower_lobe", "waist", "upper_lobe", "shoulder", None, None)
        return pts, names

    def outline(self, z):
        """The cross-section as a closed polyline from the bottom middle, its arc lengths, and
        the arc-length fraction of each named control point on the right (+x) side."""
        key = round(self.clamp(z) * 2) / 2
        if key in self._cache:
            return self._cache[key]
        from scipy.interpolate import CubicSpline
        right, names = self.control(key)
        loop = right + [(-x, y) for x, y in right[-2:0:-1]] + [right[0]]
        loop = np.array(loop)
        t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))]
        spline = CubicSpline(t, loop, bc_type="periodic")
        tt = np.linspace(0, t[-1], 241)
        pts = spline(tt)
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        named = {}
        for (cx, cy), n, tk in zip(right, names, t[:len(right)]):
            if n:
                named[n] = np.interp(tk, tt, d) / d[-1]
        self._cache[key] = (pts, d, named)
        return self._cache[key]

    def at(self, z, s):
        """3D point at length z, fraction s around (wrapping)."""
        pts, d, _ = self.outline(z)
        u = (s % 1.0) * d[-1]
        return np.array([np.interp(u, d, pts[:, 0]), np.interp(u, d, pts[:, 1]), z])

    def named(self, z, name, side=1):
        """3D point of a named control point at z, on the right (+1) or left (-1) side."""
        _, _, nm = self.outline(z)
        s = nm[name] if side > 0 else 1.0 - nm[name]
        return self.at(z, s)

    def perimeter(self, z):
        return self.outline(z)[1][-1]


def seam_bosses(L, tube):
    """(x, y) of the four Pin bosses: at the shelf edges and the shoulders at the Seam, moved a
    little inward so the boss sits inside the cage's skin."""
    z = L.length / 2
    out = []
    for name, inward in (("shelf", (2.5, 2.0)), ("shoulder", (2.5, -2.5))):
        for sx in (-1, 1):
            p = tube.named(z, name, sx)
            out.append((p[0] - sx * inward[0], max(p[1] + inward[1], joint.BOSS_R + 0.3)))
    return out


def off_seam(points, L, tube, clear=SEAM_CLEAR):
    """True if a Strut through `points` stays clear of the Seam plane, except where it runs
    into a Pin boss (the bosses, Rails, Side guides and waist lines are the Seam's only
    contacts)."""
    z_s = L.length / 2
    pts = np.atleast_2d(np.asarray(points, float))
    near = np.abs(pts[:, 2] - z_s) < clear
    if not near.any():
        return bool(np.all(np.sign(pts[:, 2] - z_s) == np.sign(pts[0, 2] - z_s)))
    bosses = np.array(seam_bosses(L, tube))
    d = np.min(np.hypot(pts[near, 0, None] - bosses[None, :, 0],
                        pts[near, 1, None] - bosses[None, :, 1]), axis=1)
    return bool(np.all(d < joint.BOSS_R + 1.0))


def in_opening(p, L, tube):
    """Holes in the Lattice: the loading opening over the Upper level's front, and the sides
    beside the front egg, for fingers."""
    x, y, z = p
    z_open = L.z_front - 75.0
    if z > z_open and y > tube.top(z) - 11 and abs(x) < 0.72 * tube.half_width(z):
        return True
    y_low_top = float(L.lower_y(L.z_front)) + XL["diameter"] / 2 + 3
    if z > L.z_front - 12 and abs(x) > 20 and tube.bottom(z) + 8 < y < y_low_top:
        return True
    return False


# --- The wrapped Lattice ---

def wrapped_voronoi(L, tube, rng):
    """Voronoi network on the tube, in (z, s); returns nodes (n, 2) in (z, s) and edges."""
    P0 = tube.perimeter(L.length / 2)
    z0, z1 = BACK_Z, L.length - 2.5
    sp = SPACING
    seeds = []
    for _ in range(20000):
        p = np.array([rng.uniform(z0 - sp, z1 + sp) / ANISOTROPY, rng.uniform(0, P0)])
        if seeds:
            q = np.array(seeds)
            dd = np.abs(q - p)
            dd[:, 1] = np.minimum(dd[:, 1], P0 - dd[:, 1])
            if np.min(np.hypot(dd[:, 0], dd[:, 1])) < sp:
                continue
        seeds.append(p)
    seeds = np.array(seeds)
    tiled = np.concatenate([seeds + [0, k * P0] for k in (-1, 0, 1)])
    vor = Voronoi(tiled)
    V = vor.vertices.copy()
    V[:, 0] *= ANISOTROPY
    nodes, index, edges, on_rim = [], {}, [], []

    def node(zs, rim=False):
        key = (round(zs[0], 2), round(zs[1] % P0, 2))
        if key not in index:
            index[key] = len(nodes)
            nodes.append(np.array([zs[0], (zs[1] % P0) / P0]))
            on_rim.append(rim)
        return index[key]
    for i, j in vor.ridge_vertices:
        if i < 0 or j < 0:
            continue
        a, b = V[i].copy(), V[j].copy()
        mid_s = (a[1] + b[1]) / 2
        if not (0 <= mid_s < P0):
            continue
        za, zb = a[0], b[0]
        if (za < z0 and zb < z0) or (za > z1 and zb > z1):
            continue
        ra = rb = False
        for lim in (z0, z1):        # clip to the back and front rings
            for pt, other in ((a, b), (b, a)):
                if (pt[0] < z0 and lim == z0) or (pt[0] > z1 and lim == z1):
                    t = (lim - other[0]) / (pt[0] - other[0])
                    pt[:] = other + t * (pt - other)
        ra, rb = abs(a[0] - z0) < 1e-6 or abs(a[0] - z1) < 1e-6, \
            abs(b[0] - z0) < 1e-6 or abs(b[0] - z1) < 1e-6
        edges.append((node(a, ra), node(b, rb)))
    return np.array(nodes), edges, np.array(on_rim)


def strut_path(tube, a, b, rng, curve=CURVE, n=6):
    """A curved Strut from a to b ((z, s) on the tube), bowing around the tube (in s)."""
    a, b = np.array(a, float), np.array(b, float)
    if abs(b[1] - a[1]) > 0.5:            # unwrap across s = 0
        b[1] += -1.0 if b[1] > a[1] else 1.0
    P = tube.perimeter((a[0] + b[0]) / 2)
    length = math.hypot(b[0] - a[0], (b[1] - a[1]) * P)
    bow = rng.uniform(-1, 1) * curve * length
    # sideways in (z, s-length) space
    dz, ds = b[0] - a[0], (b[1] - a[1]) * P
    nz, ns = -ds / (length or 1), dz / (length or 1)
    ctrl = (a + b) / 2 + np.array([nz * bow, ns * bow / P])
    t = np.linspace(0, 1, n)[:, None]
    q = (1 - t) ** 2 * a + 2 * (1 - t) * t * ctrl + t ** 2 * b
    return np.array([tube.at(z, s) for z, s in q])


def make_printable(tube, nodes, edges, on_rim, rounds=60):
    """Nudge free Nodes along z until every edge passes ok_angle (straight-line test)."""
    nodes = nodes.copy()
    to3 = lambda k: tube.at(*nodes[k])
    for _ in range(rounds):
        moved = False
        for i, j in edges:
            a, b = to3(i), to3(j)
            if ok_angle(a, b):
                continue
            lat = math.hypot(*(b - a)[:2])
            dz = b[2] - a[2]
            if lat <= MAX_BRIDGE and abs(dz) < 0.4 * lat:
                k, o = (j, i) if not on_rim[j] else (i, j)
                if not on_rim[k]:
                    nodes[k][0] = nodes[o][0]
                    moved = True
                continue
            target = lat / math.tan(math.radians(SHORT_ANGLE - 2)) if lat <= SHORT * 0.8 else \
                lat / math.tan(math.radians(43))
            need = target - abs(dz)
            sgn = 1 if dz >= 0 else -1
            if not on_rim[j]:
                nodes[j][0] += sgn * need
            elif not on_rim[i]:
                nodes[i][0] -= sgn * need
            else:
                continue
            moved = True
        if not moved:
            break
    return nodes


def prop_support(nodes, edges, base_ok):
    """Nodes that print: base_ok ones, and any reached by an edge from a lower (or level)
    printing Node."""
    ok = base_ok.copy()
    adj = [[] for _ in nodes]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    order = np.argsort(nodes[:, 0])
    for _ in range(3):
        for k in order:
            if ok[k]:
                continue
            for m in adj[k]:
                if ok[m] and nodes[m][0] <= nodes[k][0] + 0.5:
                    ok[k] = True
                    break
    return ok


def prop_and_prop(tube, nodes, edges, base_ok, L, eggs):
    """Make every used Node print: where one has nothing below it, add an edge down to the
    nearest printing Node within reach; drop what still can't print."""
    edges = list(edges)
    for _ in range(4):
        ok = prop_support(nodes, edges, base_ok)
        used = set(i for e in edges for i in e)
        bad = [k for k in sorted(used, key=lambda k: nodes[k][0]) if not ok[k]]
        if not bad:
            break
        p3 = np.array([tube.at(*n) for n in nodes])
        for k in bad:
            if ok[k]:
                continue
            cand = np.nonzero(ok & (nodes[:, 0] < nodes[k][0] - 1))[0]
            if len(cand) == 0:
                continue
            d = np.linalg.norm(p3[cand] - p3[k], axis=1)
            for m in cand[np.argsort(d)][:12]:
                if np.linalg.norm(p3[m] - p3[k]) > 45:
                    break
                if not ok_angle(p3[m], p3[k]) or in_opening((p3[m] + p3[k]) / 2, L, tube):
                    continue
                if segment_hits_egg(p3[m], p3[k], R_END, eggs):
                    continue
                edges.append((m, k))
                ok[k] = True
                break
    ok = prop_support(nodes, edges, base_ok)
    return [(i, j) for i, j in edges if ok[i] and ok[j]]


def route_to_bosses(L, tube, nodes, edges, on_rim):
    """Only the Pin bosses, Rails, Side guides and waist lines cross the Seam. Nodes near the
    Seam move off
    it, and each Lattice edge that crossed it becomes two, one each side, running into the
    nearest boss: the Struts gather into the joints like tendons."""
    z_s = L.length / 2
    off = SEAM_CLEAR + 2.0
    nodes = nodes.copy()
    for k in range(len(nodes)):
        if not on_rim[k] and abs(nodes[k][0] - z_s) < off:
            nodes[k][0] = z_s + off * (1 if nodes[k][0] >= z_s else -1)
    bosses = seam_bosses(L, tube)
    s_b = []
    for bx, by in bosses:
        pts, d, _ = tube.outline(z_s)
        k = np.argmin(np.hypot(pts[:, 0] - bx, pts[:, 1] - by))
        s_b.append(d[k] / d[-1])
    anchor = {}
    extra = []
    for b, sb in enumerate(s_b):
        for side in (-1, 1):
            anchor[(b, side)] = len(nodes) + len(extra)
            extra.append((z_s + side * off, sb))
    nodes = np.vstack([nodes, np.array(extra)])
    on_rim = np.r_[on_rim, np.ones(len(extra), bool)]
    out = set()
    for i, j in edges:
        zi, zj = nodes[i][0], nodes[j][0]
        if (zi - z_s) * (zj - z_s) >= 0:
            out.add((i, j))
            continue
        t = (z_s - zi) / (zj - zi)
        si, sj = nodes[i][1], nodes[j][1]
        if abs(sj - si) > 0.5:
            sj += -1.0 if sj > si else 1.0
        sc = (si + t * (sj - si)) % 1.0
        b = int(np.argmin([min(abs(sc - x), 1 - abs(sc - x)) for x in s_b]))
        for k, z in ((i, zi), (j, zj)):
            side = 1 if z > z_s else -1
            out.add((k, anchor[(b, side)]))
    return nodes, sorted(out), on_rim


def lattice(g, L, tube, rng, eggs, fixed_lines):
    nodes, edges, on_rim = wrapped_voronoi(L, tube, rng)
    nodes = make_printable(tube, nodes, edges, on_rim)
    nodes, edges, on_rim = route_to_bosses(L, tube, nodes, edges, on_rim)
    keep = []
    why = {"angle": 0, "opening": 0, "short": 0}
    for i, j in edges:
        a, b = tube.at(*nodes[i]), tube.at(*nodes[j])
        if np.linalg.norm(b - a) < 3:
            why["short"] += 1
            continue
        if not ok_angle(a, b):
            why["angle"] += 1
            continue
        if in_opening(a, L, tube) or in_opening(b, L, tube) or in_opening((a + b) / 2, L, tube):
            why["opening"] += 1
            continue
        keep.append((i, j))
    n_before = len(keep)
    # Nodes resting on a frame line, Side guide or Rail print: those run unbroken.
    p3 = np.array([tube.at(*n) for n in nodes])
    near = np.zeros(len(nodes), bool)
    for line in fixed_lines:
        d = np.min(np.linalg.norm(p3[:, None, :] - line[None, :, :], axis=2), axis=1)
        near |= d < 4.5
    keep = prop_and_prop(tube, nodes, keep, on_rim | near, L, eggs)
    why["unsupported"] = n_before - len([e for e in keep if e in set(edges)])
    anchored = on_rim | near
    while True:                  # prune Struts that end at a Node with nothing else on it
        deg = np.zeros(len(nodes), int)
        for i, j in keep:
            deg[i] += 1
            deg[j] += 1
        leaf = (deg == 1) & ~anchored
        new = [(i, j) for i, j in keep if not (leaf[i] or leaf[j])]
        if len(new) == len(keep):
            break
        keep = new
    why["egg/curve"] = 0
    degree = np.zeros(len(nodes), int)
    for i, j in keep:
        path = None
        for attempt in range(3):
            p = strut_path(tube, nodes[i], nodes[j], rng, curve=CURVE * (1 - attempt / 2))
            if all(ok_angle(u, v) for u, v in zip(p[:-1], p[1:])) and not any(
                    segment_hits_egg(u, v, R_END, eggs) for u, v in zip(p[:-1], p[1:])) and \
                    not any(in_opening(q, L, tube) for q in p) and off_seam(p, L, tube):
                path = p
                break
        if path is None:
            why["egg/curve"] += 1
            continue
        g.bone(path, "lattice")
        degree[i] += 1
        degree[j] += 1
    print(f"  Lattice: {len(edges)} Voronoi edges, {int((degree > 0).sum())} Nodes used; dropped {why}")
    pts = []
    for k in np.nonzero(degree)[0]:
        p = tube.at(*nodes[k])
        pts.append(p)
        if NODE_R > 0 and degree[k] >= 3 and not on_rim[k]:
            g.nodes.append((p, NODE_R))
    return np.array(pts)


def build_graph(L, seed):
    rng = np.random.default_rng(seed)
    eggs = egg_samples(L)
    tube = Tube(L)
    g = Graph()
    Z = L.length
    zf = Z - 2.5

    # Rails and Side guides.
    z_s = Z / 2

    for s in (-1, 1):   # Rails and Side guides run straight through the Seam
        g.polyline(L.lower_rail(s), RAIL_R, "rail")
        g.polyline(L.upper_rail(s), RAIL_R, "rail")
        g.polyline(L.guide(s), GUIDE_R, "guide")

    # Frame: lines along the shelf edges, the waists and the shoulders, following the cage's
    # shape; the back ring (on the bed); the front ring round the Upper level.
    zs = np.linspace(BACK_Z, zf, 80)
    for sx in (-1, 1):
        for name in ("shoulder", "waist", "shelf"):
            line = [tube.named(z, name, sx) for z in zs]
            if name == "shelf":
                line = [q for q in line if not in_opening(q + [0, 8, 0], L, tube)]
            g.polyline(line, FRAME_R, "frame")
    back_ring = tube.outline(BACK_Z)[0]
    g.polyline([(x, y, BACK_Z) for x, y in back_ring], FRAME_R, "frame")
    front_ring = tube.outline(zf)[0]
    y_cut = float(L.lower_y(L.z_front)) + XL["diameter"] / 2 + 3
    upper = [(x, y, zf) for x, y in front_ring if y >= y_cut]
    upper = sorted(upper, key=lambda p: math.atan2(p[1] - y_cut, p[0]))
    g.polyline(upper, FRAME_R, "frame")
    # Exit cradle loop under the front egg, as the Original's.
    zc0 = L.z_front - 30
    loop = [(18 * math.sin(a), BOTTOM_Y, zc0 + (zf + 3 - zc0) * (1 - math.cos(a)) / 2)
            for a in np.linspace(-math.pi, math.pi, 40)]
    g.polyline(loop, FRAME_R, "frame")

    # V ties across the bottom, out of sight: from the bottom middle out to both shelf lines,
    # opening toward +z, one completing every TIE_SPACING mm or less, so the sides never stand
    # free much longer than that while printing.
    done = BACK_Z + BOTTOM_ROUND + 10
    while done < L.length - 15:
        right = tube.named(done, "shelf", 1)
        dz = abs(right[0]) / math.tan(math.radians(43)) * rng.uniform(1.0, 1.08)
        zc = done - dz
        if zc < z_s + SEAM_CLEAR and done > z_s - SEAM_CLEAR:
            zc = z_s + SEAM_CLEAR + 0.5            # the first front tie starts at the Seam
            done = zc + dz
        for sx in (-1, 1):
            s_end = tube.outline(done)[2]["shelf"]
            s_end = s_end if sx > 0 else -s_end
            path = strut_path(tube, (zc, 0.0), (done, s_end), rng, curve=0.04)
            if all(ok_angle(u, v) for u, v in zip(path[:-1], path[1:])) and not any(
                    segment_hits_egg(u, v, R_END, eggs) for u, v in zip(path[:-1], path[1:])):
                g.bone(path, "tie")
        done += TIE_SPACING * rng.uniform(0.85, 1.0)

    # The wrapped Lattice.
    fixed_lines = [resample(np.array(g.seg_a)[np.array(g.seg_chain) == c], 3.0)
                   for c in np.unique(g.seg_chain)]
    anchors = lattice(g, L, tube, rng, eggs, fixed_lines)
    anchors = np.concatenate([anchors] + [resample(np.array(L.guide(s)), 10.0) for s in (-1, 1)])

    # Rail supports: to the nearest Lattice or guide point that keeps them printable and clear.
    def support(a, prefer=None, max_len=60):
        a = np.asarray(a, float)
        d = np.linalg.norm(anchors - a, axis=1)
        for k in np.argsort(d):
            c = anchors[k]
            if d[k] < 8 or d[k] > max_len:
                continue
            if angle_from_z(a, c) > SUPPORT_ANGLE:
                continue
            if prefer is not None and np.sign(c[2] - a[2]) != prefer:
                continue
            if segment_hits_egg(a, c, R_END, eggs, skip_ends=6.0):
                continue
            if not off_seam(np.linspace(a, c, 12), L, tube):
                continue
            g.bone([a, (a + c) / 2, c], "support")
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
        for side in (-1, 1):                # hold the Rails' ends either side of the Seam
            z = z_s + side * 12.0
            missing += not support((s * RAIL_X, float(L.upper_rail_y(z)), z))
            missing += not support((s * RAIL_X, float(L.lower_rail_y(z)), z))
    if missing:
        print(f"  {missing} Rail supports found no anchor")

    # The Seam: four Pin bosses swelling on the shelf and shoulder frame lines. With the Rails,
    # Side guides and waist lines they are the only things that cross it; the field is cut flat there, so each contact gets
    # matching faces, and the Pin holes are cut exactly afterwards (seam_bosses).
    for (bx, by) in seam_bosses(L, tube):
        c = g.new_chain()
        h = joint.BOSS_LEN
        taper = joint.BOSS_R * 1.4           # cones down to a Strut, steeper than 45 degrees
        pts = [z_s - h - taper, z_s - h, z_s + h, z_s + h + taper]
        rad = [FRAME_R, joint.BOSS_R, joint.BOSS_R, FRAME_R]
        for k in range(3):
            g.add((bx, by, pts[k]), (bx, by, pts[k + 1]), rad[k], "boss", c, rad[k + 1])

    # Back ring rungs, on the bed, from the back wall Rails out to the sides.
    back_wall = L.lower_rail(1)
    y_wall0 = back_wall[back_wall[:, 2] < BACK_Z + 0.5][:, 1].min()
    for y in np.arange(y_wall0 + 8, tube.top(BACK_Z) - 6, 22.0):
        for sx in (-1, 1):
            ring_x = back_ring[:, 0][(back_ring[:, 0] * sx > 0)]
            ring_y = back_ring[:, 1][(back_ring[:, 0] * sx > 0)]
            xw = float(np.abs(ring_x[np.argmin(np.abs(ring_y - y))]))
            a, b = (sx * RAIL_X, y, BACK_Z), (sx * xw, y, BACK_Z)
            if not segment_hits_egg(a, b, FRAME_R, eggs, skip_ends=3.0):
                g.add(a, b, FRAME_R, "back")

    fixed, left = tie_off(g, L, eggs)
    print(f"  tied off {fixed} dangling ends, {left} left")
    return g


# --- Loose ends ---

def chain_ends(g):
    A, B, R, K = g.arrays()
    ch = np.array(g.seg_chain)
    ends = []
    for cid in np.unique(ch):
        idx = np.nonzero(ch == cid)[0]
        pts = np.concatenate([A[idx], B[idx]])
        keys = [tuple(np.round(p, 4)) for p in pts]
        for p, k in zip(pts, keys):
            if keys.count(k) == 1:
                ends.append((cid, p, R[idx[0]], K[idx[0]]))
    return ends


def dangling(g):
    """Chain ends that touch no other Strut."""
    A, B, R, K = g.arrays()
    ch = np.array(g.seg_chain)
    out = []
    for cid, p, r, kind in chain_ends(g):
        other = ch != cid
        a, b, rr = A[other], B[other], R[other]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) /
                    np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
        d = np.linalg.norm(a + t[:, None] * ab - p, axis=1) - rr - r
        if d.min() > 0.5:
            out.append((cid, p, r, kind))
    return out


def tie_off(g, L, eggs):
    """Join each dangling end to the nearest point of another Strut with a printable,
    egg-clear Strut (any angle if it lies on the bed)."""
    tube = Tube(L)
    A, B, R, K = g.arrays()
    ch = np.array(g.seg_chain)
    samples, owner = [], []
    for a, b, c in zip(A, B, ch):
        for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 4) + 1)):
            samples.append(a + t * (b - a))
            owner.append(c)
    samples, owner = np.array(samples), np.array(owner)
    fixed = left = 0
    for cid, p, r, kind in dangling(g):
        if kind == "rail" and p[2] < BACK_Z + 1:
            continue
        d = np.linalg.norm(samples - p, axis=1)
        d[owner == cid] = np.inf
        done = False
        for k in np.argsort(d)[:300]:
            c = samples[k]
            if d[k] > 60 or d[k] < 1:
                continue
            on_bed = p[2] < BACK_Z + 1 and c[2] < BACK_Z + 1
            if not on_bed and not ok_angle(p, c):
                continue
            if segment_hits_egg(p, c, R_END, eggs, skip_ends=4.0):
                continue
            if not off_seam(np.linspace(p, c, 12), L, tube):
                continue
            g.bone([p, (p + c) / 2, c], "tie")
            fixed += 1
            done = True
            break
        left += not done
    return fixed, left


# --- Field and mesh ---

def capsule_field(g, lo, shape, h=VOXEL, k=BLEND_K):
    """Smooth-union distance field of the graph: f = -k log(sum exp(-d_i / k)), each chain's
    tapered segments (round cones) combined with a hard minimum first."""
    A = np.array(g.seg_a)
    B = np.array(g.seg_b)
    RA = np.array(g.seg_ra)
    RB = np.array(g.seg_rb)
    S = np.zeros(shape, np.float32)
    pad = max(RA.max(), RB.max()) + 7 * k
    axes = [lo[i] + h * np.arange(shape[i]) for i in range(3)]

    def accumulate(dist_fn, bmin, bmax):
        i0 = np.maximum(np.floor((bmin - lo) / h).astype(int), 0)
        i1 = np.minimum(np.ceil((bmax - lo) / h).astype(int) + 1, shape)
        if np.any(i1 <= i0):
            return
        X = axes[0][i0[0]:i1[0], None, None]
        Y = axes[1][None, i0[1]:i1[1], None]
        Zc = axes[2][None, None, i0[2]:i1[2]]
        d = dist_fn(X, Y, Zc)
        S[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] += np.exp(-d / k).astype(np.float32)

    def round_cone(X, Y, Zc, a, b, r1, r2):
        """Distance to a capsule whose radius goes from r1 at a to r2 at b (iq's round cone)."""
        ba = b - a
        l2 = float(ba @ ba)
        rr = r1 - r2
        a2 = l2 - rr * rr
        il2 = 1.0 / l2
        px, py, pz = X - a[0], Y - a[1], Zc - a[2]
        y = px * ba[0] + py * ba[1] + pz * ba[2]
        z = y - l2
        qx, qy, qz = px * l2 - ba[0] * y, py * l2 - ba[1] * y, pz * l2 - ba[2] * y
        x2 = qx * qx + qy * qy + qz * qz
        y2 = y * y * l2
        z2 = z * z * l2
        kk = math.copysign(1, rr) * rr * rr * x2
        d_mid = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
        d_b = np.sqrt(x2 + z2) * il2 - r2
        d_a = np.sqrt(x2 + y2) * il2 - r1
        return np.where(np.sign(z) * a2 * z2 > kk, d_b, np.where(np.sign(y) * a2 * y2 < kk, d_a, d_mid))

    chain_ids = np.array(g.seg_chain)
    for cid in np.unique(chain_ids):
        idx = [i for i in np.nonzero(chain_ids == cid)[0] if np.linalg.norm(B[i] - A[i]) > 1e-6]
        if not idx:
            continue
        pts = np.concatenate([A[idx], B[idx]])
        c0 = np.maximum(np.floor((pts.min(0) - pad - lo) / h).astype(int), 0)
        c1 = np.minimum(np.ceil((pts.max(0) + pad - lo) / h).astype(int) + 1, shape)
        if np.any(c1 <= c0):
            continue
        best = np.full(tuple(c1 - c0), np.inf, np.float32)
        for i in idx:
            a, b, r1, r2 = A[i], B[i], RA[i], RB[i]
            i0 = np.maximum(np.floor((np.minimum(a, b) - pad - lo) / h).astype(int), c0)
            i1 = np.minimum(np.ceil((np.maximum(a, b) + pad - lo) / h).astype(int) + 1, c1)
            if np.any(i1 <= i0):
                continue
            X = axes[0][i0[0]:i1[0], None, None]
            Y = axes[1][None, i0[1]:i1[1], None]
            Zc = axes[2][None, None, i0[2]:i1[2]]
            d = round_cone(X, Y, Zc, a, b, r1, r2)
            sl = tuple(slice(i0[k] - c0[k], i1[k] - c0[k]) for k in range(3))
            np.minimum(best[sl], d, out=best[sl])
        S[c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]] += np.exp(-best / k)
    for c, r in g.nodes:
        c = np.asarray(c)

        def dist(X, Y, Zc, c=c, r=r):
            return np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Zc - c[2]) ** 2) - r
        accumulate(dist, c - r - 7 * k, c + r + 7 * k)
    with np.errstate(divide="ignore"):
        f = -k * np.log(S)
    f[~np.isfinite(f)] = 50.0
    return f, axes


def half_mesh(g, L, which, h=VOXEL):
    """Mesh one half: 'back' (z from 0 to the Seam) or 'front' (Seam to the front)."""
    zs = L.length / 2
    z0, z1 = (0.0, zs) if which == "back" else (zs, float(L.length) + 4)
    lo = np.array([-SHELL_X - 16, -1.0, z0 - 1.0])
    hi = np.array([SHELL_X + 16, L.height + ARCH + 10, z1 + 1.0])
    shape = tuple(np.ceil((hi - lo) / h).astype(int) + 1)
    f, axes = capsule_field(g, lo, shape, h)
    X, Y, Zc = np.meshgrid(*axes, indexing="ij", sparse=True)
    f = np.maximum(f, -Y)                        # shelf
    f = np.maximum(f, z0 - Zc)                   # bed / Seam
    f = np.maximum(f, Zc - z1)
    f[[0, -1], :, :] = 50.0
    f[:, [0, -1], :] = 50.0
    f[:, :, [0, -1]] = 50.0
    verts, faces, _, _ = marching_cubes(f, 0.0, spacing=(h, h, h))
    verts += lo
    m = trimesh.Trimesh(verts, faces, process=True)
    if m.volume < 0:
        m.invert()
    import fast_simplification
    v2, f2 = fast_simplification.simplify(m.vertices, m.faces, target_reduction=0.75)
    m = trimesh.Trimesh(v2, f2, process=True)
    if m.volume < 0:
        m.invert()
    body = mf.Manifold(mf.Mesh(vert_properties=np.asarray(m.vertices, np.float32),
                               tri_verts=np.asarray(m.faces, np.uint32)))
    assert body.status() == mf.Error.NoError, body.status()
    cut = joint.holes(seam_bosses(L, Tube(L)), zs, -1 if which == "back" else 1)
    return body - cut


def build(seed=1, write=True):
    L = Layout()
    t0 = time.time()
    g = build_graph(L, seed)
    A, B, R, K = g.arrays()
    print(f"seed {seed}: {len(A)} segments, {len(g.nodes)} nodes, kinds",
          {str(k): int((K == k).sum()) for k in np.unique(K)}, f"({time.time() - t0:.1f}s)")
    halves = {}
    for which in ("back", "front"):
        t0 = time.time()
        m = half_mesh(g, L, which)
        halves[which] = m
        print(f"  {which}: {m.num_tri()} triangles, {m.volume() / 1000:.1f} cm3 "
              f"({time.time() - t0:.1f}s)")
        if write:
            BUILD.mkdir(exist_ok=True)
            joint.to_trimesh(m).export(BUILD / f"grown_{seed}_{which}.stl")
    if write:
        joint.to_trimesh(halves["back"] + halves["front"]).export(BUILD / f"grown_{seed}.stl")
    return g, halves


if __name__ == "__main__":
    for s in [int(a) for a in sys.argv[1:]] or [1]:
        build(s)
