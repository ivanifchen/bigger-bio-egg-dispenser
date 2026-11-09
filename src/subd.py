"""The skin as a subdivision surface: picture frames round the cells, thickened, Catmull-Clark.

The method Rhino/Grasshopper's Weaverbird made common for organic lattices: each cell of the
chart (a polygon) keeps a band of quads along its edges, a "picture frame", and its middle is a
hole; edges on the chart's boundary (rims, openings) get a band outward as well. The quad mesh
is mapped onto the Tube, thickened into a closed shell and subdivided: Catmull-Clark rounds it
into Struts that thin between Nodes and fork smoothly at them.
"""

import math

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import polygonize, unary_union


# --- Catmull-Clark ---

def catmull_clark(V, F):
    """One Catmull-Clark step on a quad mesh (V (n, 3), F (m, 4)); edges with one face keep
    the boundary rule."""
    nV, nF = len(V), len(F)
    fp = V[F].mean(axis=1)
    e = np.stack([F, np.roll(F, -1, axis=1)], axis=2).reshape(-1, 2)          # (4m, 2)
    es = np.sort(e, axis=1)
    uniq, inv = np.unique(es, axis=0, return_inverse=True)
    inv = inv.ravel()
    nE = len(uniq)
    face_of = np.repeat(np.arange(nF), 4)
    cnt = np.bincount(inv, minlength=nE)
    fsum = np.zeros((nE, 3))
    np.add.at(fsum, inv, fp[face_of])
    mid = V[uniq].mean(axis=1)
    ep = np.where((cnt >= 2)[:, None],
                  (V[uniq].sum(axis=1) + fsum) / (2 + np.maximum(cnt, 1))[:, None], mid)
    # vertex points
    val = np.bincount(uniq.ravel(), minlength=nV).astype(float)
    nf = np.bincount(F.ravel(), minlength=nV).astype(float)
    Fs = np.zeros((nV, 3))
    np.add.at(Fs, F.ravel(), np.repeat(fp, 4, axis=0))
    Rs = np.zeros((nV, 3))
    np.add.at(Rs, uniq[:, 0], mid)
    np.add.at(Rs, uniq[:, 1], mid)
    n = val
    Favg = Fs / np.maximum(nf, 1)[:, None]
    Ravg = Rs / np.maximum(n, 1)[:, None]
    newV = (Favg + 2 * Ravg + (n - 3)[:, None] * V) / np.maximum(n, 1)[:, None]
    # boundary vertices: average of the boundary edge midpoints and the vertex
    bnd = cnt == 1
    if bnd.any():
        bs = np.zeros((nV, 3))
        bc = np.zeros(nV)
        bu = uniq[bnd]
        bm = mid[bnd]
        np.add.at(bs, bu[:, 0], bm)
        np.add.at(bs, bu[:, 1], bm)
        np.add.at(bc, bu[:, 0], 1)
        np.add.at(bc, bu[:, 1], 1)
        isb = bc > 0
        newV[isb] = (bs[isb] / bc[isb, None] + V[isb]) / 2
    # new faces
    ei = inv.reshape(nF, 4)                    # edge (k, k+1) of each face
    vo = 0
    eo = nV
    fo = nV + nE
    P = np.vstack([newV, ep, fp])
    a, b, c, d = F.T
    eab, ebc, ecd, eda = ei.T
    f = np.arange(nF)
    newF = np.concatenate([
        np.stack([a, eo + eab, fo + f, eo + eda], 1),
        np.stack([b, eo + ebc, fo + f, eo + eab], 1),
        np.stack([c, eo + ecd, fo + f, eo + ebc], 1),
        np.stack([d, eo + eda, fo + f, eo + ecd], 1)])
    return P, newF


# --- The cells as a planar quad mesh on the chart ---

def planar_faces(cells, veins, domain, P0, min_area=12.0, tol=0.0):
    """The chart cut by the cells' edges and the veins, inside the domain: faces sharing exact
    vertices, for u in [0, P0) (faces crossing u = 0 or P0 are kept once)."""
    tile = lambda g: unary_union([shapely.affinity.translate(g, 0, k * P0) for k in (-1, 0, 1)])
    lines = [c.boundary for c in cells if not c.is_empty]
    lines += [tile(MultiLineString([v]) if isinstance(v, LineString) else v) for v in veins]
    lines.append(domain.boundary)
    net = shapely.set_precision(unary_union(lines), 0.01)
    if tol > 0:
        # a coarse cage: each line between junctions down to a few points (junctions stay)
        merged = shapely.line_merge(net)
        parts = getattr(merged, "geoms", [merged])
        net = unary_union([p.simplify(tol, preserve_topology=False) for p in parts])
    faces = []
    for f in polygonize(net):
        if f.area < 1e-3:
            continue
        rp = f.representative_point()
        if not domain.contains(rp):
            continue
        c = f.centroid
        if not (0 <= c.y < P0):
            continue
        faces.append(f)
    # merge slivers into the neighbour they share most edge with
    faces = merge_slivers(faces, min_area)
    return faces


def merge_slivers(faces, min_area):
    faces = list(faces)
    changed = True
    while changed:
        changed = False
        for i, f in enumerate(faces):
            thin = f.area < min_area or f.area / f.length < 0.9
            if not thin:
                continue
            best, share = None, 0.0
            for j, g in enumerate(faces):
                if j == i:
                    continue
                s = f.boundary.intersection(g.boundary).length
                if s > share:
                    best, share = j, s
            if best is None or share < 1e-6:
                continue
            m = unary_union([faces[best], f])
            if isinstance(m, Polygon):
                faces[best] = shapely.set_precision(m, 0.01)
                del faces[i]
                changed = True
                break
    return faces


def inset_loop(xy, w, max_frac=0.42):
    """A closed loop (CCW, no repeat) moved inward by w (mitred), each point at most max_frac
    of the way to the centroid."""
    n = len(xy)
    c = xy.mean(axis=0)
    prev = xy[np.arange(n) - 1]
    nxt = xy[(np.arange(n) + 1) % n]
    d1 = xy - prev
    d2 = nxt - xy
    d1 /= np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-9)
    d2 /= np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-9)
    n1 = np.column_stack([-d1[:, 1], d1[:, 0]])      # left normals: inward for CCW
    n2 = np.column_stack([-d2[:, 1], d2[:, 0]])
    m = n1 + n2
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    cosh = np.maximum(np.einsum("ij,ij->i", m, n1), 0.35)
    q = xy + m * (w / cosh)[:, None]
    # never past max_frac of the way to the centroid
    to_c = c - xy
    dist = np.linalg.norm(to_c, axis=1)
    step = np.linalg.norm(q - xy, axis=1)
    lim = max_frac * dist
    over = step > lim
    q[over] = xy[over] + (q[over] - xy[over]) * (lim[over] / step[over])[:, None]
    return q


def frame_mesh(faces, w, P0, key_tol=0.02):
    """Picture frames: per face a band of quads of width w inside its edges; plus a band of
    width w outside every boundary edge (an edge with only one face). Returns chart vertices
    (n, 2), quads (m, 4), and which vertices lie on hole loops (inner) or outer loops."""
    verts, index = [], {}

    def vid(p):
        k = (round(p[0] / key_tol), round((p[1] % P0) / key_tol) % round(P0 / key_tol))
        if k not in index:
            index[k] = len(verts)
            verts.append((p[0], p[1]))
        return index[k]

    quads = []
    edge_faces = {}
    loops = []
    for f in faces:
        f = shapely.geometry.polygon.orient(f, 1.0)
        xy = np.asarray(f.exterior.coords)[:-1]
        # drop repeated points
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.05]
        xy = xy[keep]
        if len(xy) < 3:
            continue
        P = [vid(p) for p in xy]
        Q = inset_loop(xy, w)
        Qi = list(range(len(verts), len(verts) + len(Q)))
        verts.extend(map(tuple, Q))
        n = len(P)
        for i in range(n):
            j = (i + 1) % n
            quads.append((P[i], P[j], Qi[j], Qi[i]))
            edge_faces.setdefault(tuple(sorted((P[i], P[j]))), []).append((P[i], P[j]))
        loops.append(Qi)
    V = np.array(verts, float)
    # boundary edges: one face. Band outward (to the right of the face's CCW edge).
    bnd = [e[0] for e in edge_faces.values() if len(e) == 1]
    if bnd:
        normals = {}
        for a, b in bnd:
            d = V[b] - V[a]
            if d[1] > P0 / 2:
                d[1] -= P0
            if d[1] < -P0 / 2:
                d[1] += P0
            d /= max(np.linalg.norm(d), 1e-9)
            out = np.array([d[1], -d[0]])
            for k in (a, b):
                normals[k] = normals.get(k, 0) + out
        O = {}
        for k, nv in normals.items():
            nv = nv / max(np.linalg.norm(nv), 1e-9)
            O[k] = len(V) + len(O)
        extra = np.array([V[k] + w * normals[k] / max(np.linalg.norm(normals[k]), 1e-9)
                          for k in O])
        V = np.vstack([V, extra])
        for a, b in bnd:
            quads.append((b, a, O[a], O[b]))
    return V, np.array(quads, int), np.array(sorted(set(index.values())), int)


# --- Thicken onto the Tube ---

def chart_points(chart, V):
    """3D points of chart (v, u) coordinates, from the chart's tables (vectorized)."""
    u = np.mod(V[:, 1], chart.P0)
    z = chart.z_of(V[:, 0], u)
    ci = [(z - chart.zg[0]) / chart.dz, u / chart.du]
    x = ndimage.map_coordinates(chart.Sx, ci, order=1, mode="nearest")
    y = ndimage.map_coordinates(chart.Sy, ci, order=1, mode="nearest")
    return np.column_stack([x, y, z])


def thicken(chart, V2, Q, h):
    """The chart quad mesh on the Tube, offset +-h along the normal and closed along its
    boundary loops: a closed quad mesh."""
    P = chart_points(chart, V2)
    e = 0.3
    Pv = chart_points(chart, V2 + [e, 0]) - chart_points(chart, V2 - [e, 0])
    Pu = chart_points(chart, V2 + [0, e]) - chart_points(chart, V2 - [0, e])
    N = np.cross(Pu, Pv)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    mid = np.array([0.0, np.median(P[:, 1])])
    flip = np.einsum("ij,ij->i", N[:, :2], P[:, :2] - mid) < 0
    N[flip] *= -1
    n = len(P)
    top = P + h * N
    bot = P - h * N
    Vs = np.vstack([top, bot])
    # orientation: chart quads are CCW in (v, u); pick the order whose normal points out
    q0 = Q[0]
    nrm = np.cross(top[q0[1]] - top[q0[0]], top[q0[3]] - top[q0[0]])
    if nrm @ N[q0[0]] < 0:
        Q = Q[:, ::-1]
    F = [Q, Q[:, ::-1] + n]
    # walls along boundary edges of the chart mesh
    e = np.stack([Q, np.roll(Q, -1, axis=1)], axis=2).reshape(-1, 2)
    es = np.sort(e, axis=1)
    uniq, inv, cnt = np.unique(es, axis=0, return_inverse=True, return_counts=True)
    once = cnt[inv.ravel()] == 1
    be = e[once]
    F.append(np.column_stack([be[:, 1], be[:, 0], be[:, 0] + n, be[:, 1] + n]))
    return Vs, np.vstack(F)


def shell_on_cylinder(V2, Q, h, P0):
    """The chart quad mesh as a closed shell on a cylinder (x = v, angle = u), +-h radially:
    smoothing here keeps the Tube's own shape; u wraps without a seam."""
    R = P0 / (2 * math.pi)
    th = V2[:, 1] / R
    n = len(V2)
    top = np.column_stack([V2[:, 0], (R + h) * np.cos(th), (R + h) * np.sin(th)])
    bot = np.column_stack([V2[:, 0], (R - h) * np.cos(th), (R - h) * np.sin(th)])
    F = [Q, Q[:, ::-1] + n]
    e = np.stack([Q, np.roll(Q, -1, axis=1)], axis=2).reshape(-1, 2)
    es = np.sort(e, axis=1)
    uniq, inv, cnt = np.unique(es, axis=0, return_inverse=True, return_counts=True)
    be = e[cnt[inv.ravel()] == 1]
    F.append(np.column_stack([be[:, 1], be[:, 0], be[:, 0] + n, be[:, 1] + n]))
    return np.vstack([top, bot]), np.vstack(F), R


def from_cylinder(chart, C, R):
    """Cylinder points back onto the Tube: (v, u, n) -> point(v, u) + n * normal."""
    v = C[:, 0]
    th = np.arctan2(C[:, 2], C[:, 1])
    u = np.mod(th * R, chart.P0)
    nn = np.hypot(C[:, 1], C[:, 2]) - R
    V2 = np.column_stack([v, u])
    P = chart_points(chart, V2)
    e = 0.3
    Pv = chart_points(chart, V2 + [e, 0]) - chart_points(chart, V2 - [e, 0])
    Pu = chart_points(chart, V2 + [0, e]) - chart_points(chart, V2 - [0, e])
    N = np.cross(Pu, Pv)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    mid = np.array([0.0, np.median(P[:, 1])])
    flip = np.einsum("ij,ij->i", N[:, :2], P[:, :2] - mid) < 0
    N[flip] *= -1
    return P + nn[:, None] * N


def skin_mesh(chart, cells, veins, domain, w, h, levels=3, tol=0.0):
    faces = planar_faces(cells, veins, domain, chart.P0, tol=tol)
    V2, Q, p_ids = frame_mesh(faces, w, chart.P0)
    # unwrap each vertex's u to one turn so the cylinder closes the seam
    V2 = V2.copy()
    V2[:, 1] = np.mod(V2[:, 1], chart.P0)
    C, F, R = shell_on_cylinder(V2, Q, h, chart.P0)
    # outward (radial) faces: make the first top quad's normal point out
    q = F[0]
    nrm = np.cross(C[q[1]] - C[q[0]], C[q[3]] - C[q[0]])
    if nrm @ np.r_[0, C[q[0], 1:]] < 0:
        F = F[:, ::-1]
    for _ in range(levels):
        C, F = catmull_clark(C, F)
    V = from_cylinder(chart, C, R)
    tris = np.vstack([F[:, [0, 1, 2]], F[:, [0, 2, 3]]])
    anchors = chart_points(chart, V2[p_ids])          # Strut centre lines, for supports
    return V, tris, anchors, len(faces)
