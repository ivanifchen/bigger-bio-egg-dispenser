"""The egg path every prototype shares: where eggs sit, the Rails under them and the Side guides
beside them, and how long the Dispenser must be to hold its eggs.

Use frame, in mm: x across the Dispenser (0 on its centre line), y up from the fridge shelf,
z along it from the back face (0) to the front. Each half prints with z vertical.

An egg lies crosswise (long axis along x) and rolls on two Rails at x = +-RAIL_X. Seen from the
side (the y-z plane), egg centres follow the Lower level from the Exit cradle back to the Turn,
stack in a short column against the back wall, and follow the Upper level from the Turn
forward to the Loading lip.
"""

import math
from dataclasses import dataclass, field

import numpy as np

# --- Eggs (US sizes) ---
LARGE = dict(length=57.0, diameter=44.0)
XL = dict(length=62.0, diameter=47.0)
EGG_PITCH = 47.0          # along a Level, centre to centre: an XL egg's diameter
EGGS = 16                 # total, both Levels

# --- Egg path ---
SLOPE_DEG = 2.2           # both Levels (ADR 0003)
RAIL_R = 2.15             # the Original's 4.3 mm Rails
RAIL_X = 13.0             # Rails at x = +-13, 26 mm apart, as the Original
CHANNEL = 64.0            # between the Side guides' inner faces (ADR 0003)
GUIDE_R = 2.5
GUIDE_X = CHANNEL / 2 + GUIDE_R
LOWER_RAIL_FRONT_Y = 7.0  # Lower level Rail centre at the Exit cradle
LEVEL_GAP_BACK = 55.0     # egg-centre to egg-centre between Levels at the back: an XL egg's
                          # diameter plus a ~8 mm step down onto the egg below in the Turn
BACK_WALL_Z = RAIL_R      # the back Rails lie on the bed
TURN_ARC_R = 25.0         # Rail bend radius where the Lower level meets the back wall
CRADLE_ARC_R = 45.0       # Exit cradle: the Rails curve up this radius, through 45 degrees
LIP_ARC_R = 40.0          # Loading lip: same, on the Upper level
FRONT_MARGIN = 8.0        # front egg's nose to the front face

# --- Cage ---
STRUT_R = 2.2             # 4.4 mm Struts, a little under the Original's ~5 mm, to keep print time down
NODE_R = 3.2              # Nodes swell to ~6.5 mm, plus the blend
SHELL_X = GUIDE_X         # side panels lie in the Side guides' plane
TOP_CLEAR = 3.0           # XL egg top to the top panel's Struts
BOTTOM_Y = 1.8            # bottom panel Strut centre; the shelf cut flattens a ~3.5 mm foot


def section_radius(egg, x):
    """Radius of an egg's round cross-section at distance x from its middle (ellipsoid)."""
    a, b = egg["length"] / 2, egg["diameter"] / 2
    return b * math.sqrt(max(0.0, 1 - (x / a) ** 2))


def centre_offset(egg):
    """How far an egg's centre sits from a Rail's centre line, square to the Rail."""
    return section_radius(egg, RAIL_X) + RAIL_R


@dataclass
class Layout:
    eggs: int = EGGS
    slope_deg: float = SLOPE_DEG
    pitch: float = EGG_PITCH
    t: float = field(init=False)
    c_off: float = field(init=False)         # Large egg centre above its Rails
    c_off_xl: float = field(init=False)
    z_column: float = field(init=False)      # egg centres in the Turn's column
    z_front: float = field(init=False)       # front egg centres (both Levels)
    length: float = field(init=False)

    def __post_init__(self):
        self.t = math.tan(math.radians(self.slope_deg))
        self.c_off = centre_offset(LARGE)
        self.c_off_xl = centre_offset(XL)
        self.z_column = BACK_WALL_Z + self.c_off
        per_level = math.ceil(self.eggs / 2)
        # Lower: the column egg plus per_level-1 more at EGG_PITCH; the Upper level the same,
        # its column egg resting on the Lower one's.
        self.z_front = self.z_column + (per_level - 1) * self.pitch
        nose = self.z_front + XL["diameter"] / 2
        self.length = math.ceil(nose + FRONT_MARGIN)

    # Egg-centre heights along each Level (straight runs).
    def lower_y(self, z):
        return LOWER_RAIL_FRONT_Y + self.c_off + (self.z_front - np.asarray(z)) * self.t

    def upper_y(self, z):
        back = self.lower_y(self.z_column) + LEVEL_GAP_BACK
        return back + (np.asarray(z) - self.z_column) * self.t

    def lower_rail_y(self, z):
        return self.lower_y(z) - self.c_off

    def upper_rail_y(self, z):
        return self.upper_y(z) - self.c_off

    @property
    def upper_rail_back_z(self):
        """The Upper level's Rails stop under the second egg, so the back egg steps down
        (LEVEL_GAP_BACK - XL diameter, about 8 mm) onto the column egg below."""
        return self.z_column + 0.75 * self.pitch

    def top_y(self, z):
        """Top panel Strut centre: clears an XL egg on the Upper level."""
        return (self.upper_y(z) + (self.c_off_xl - self.c_off) + XL["diameter"] / 2
                + TOP_CLEAR + STRUT_R)

    @property
    def height(self):
        return float(self.top_y(self.length)) + STRUT_R

    # --- Polylines (each an (n, 3) array in the use frame) ---
    def lower_rail(self, side, step=2.0):
        """Exit cradle arc -> straight Lower level -> bend up the back wall -> back wall,
        rising to the Upper level's Rail height."""
        x = side * RAIL_X
        pts = []
        # Exit cradle: tangent to the straight run at the front egg, curving up 45 degrees.
        z0, y0 = self.z_front, float(self.lower_rail_y(self.z_front))
        slope = math.atan(self.t)  # the run descends toward +z at this angle
        for a in np.linspace(0, math.radians(45), 12)[::-1]:
            # circle centred above the run's end, radius CRADLE_ARC_R
            ang = -slope + a
            pts.append((z0 + CRADLE_ARC_R * math.sin(ang) - CRADLE_ARC_R * math.sin(-slope),
                        y0 + CRADLE_ARC_R * (1 - math.cos(ang)) - CRADLE_ARC_R * (1 - math.cos(-slope))))
        # Straight run back to where the bend starts.
        z_bend = BACK_WALL_Z + TURN_ARC_R
        for z in np.arange(self.z_front, z_bend, -step)[1:]:
            pts.append((z, float(self.lower_rail_y(z))))
        # Bend: quarter circle from heading -z to heading +y, ending on the back wall.
        yb = float(self.lower_rail_y(z_bend))
        cz, cy = BACK_WALL_Z + TURN_ARC_R, yb + TURN_ARC_R
        for a in np.linspace(0, math.pi / 2, 14):
            pts.append((cz - TURN_ARC_R * math.sin(a), cy - TURN_ARC_R * math.cos(a)))
        # Back wall, up to the Upper level's Rail height.
        top = float(self.upper_rail_y(self.z_column)) + 4
        for y in np.arange(cy + step, top, step):
            pts.append((BACK_WALL_Z, y))
        return np.array([(x, y, z) for z, y in pts])

    def upper_rail(self, side, step=2.0):
        """From just behind the second egg, forward to the Loading lip."""
        x = side * RAIL_X
        pts = []
        for z in np.arange(self.upper_rail_back_z, self.z_front, step):
            pts.append((z, float(self.upper_rail_y(z))))
        z0, y0 = self.z_front, float(self.upper_rail_y(self.z_front))
        slope = math.atan(self.t)  # ascends toward +z
        for a in np.linspace(0, math.radians(45) - slope, 12):
            ang = slope + a
            pts.append((z0 + LIP_ARC_R * (math.sin(ang) - math.sin(slope)),
                        y0 + LIP_ARC_R * ((1 - math.cos(ang)) - (1 - math.cos(slope)))))
        return np.array([(x, y, z) for z, y in pts])

    def guide(self, side):
        """A Side guide at egg-centre height: Lower level, a '<' up the back (each leg within
        45 degrees of z, so it prints without support), then the Upper level."""
        x = side * GUIDE_X
        knee_z = self.z_column + 0.5 * self.pitch + 8
        y_lo = float(self.lower_y(knee_z + 30))
        y_hi = float(self.upper_y(knee_z + 30))
        y_mid = (y_lo + y_hi) / 2
        back_z = max(BACK_WALL_Z + GUIDE_R, knee_z + 30 - (y_hi - y_lo) / 2 * 1.05)
        pts = []
        for z in np.arange(self.z_front - 10, knee_z + 30, -4.0):
            pts.append((z, float(self.lower_y(z))))
        pts.append((back_z, y_mid))
        for z in np.arange(knee_z + 30, self.z_front + 4, 4.0):
            pts.append((z, float(self.upper_y(z))))
        return np.array([(x, y, z) for z, y in pts])

    # --- Eggs, for capacity and clearance checks ---
    def egg_centres(self, egg=LARGE):
        """Where a full Dispenser's eggs sit (y, z), Lower level first."""
        lift = centre_offset(egg) - self.c_off
        lower = [(float(self.lower_y(z)) + lift, z)
                 for z in self.z_column + self.pitch * np.arange(math.ceil(self.eggs / 2))]
        col_top = (lower[0][0] + egg["diameter"], self.z_column)
        upper = [col_top] + [(float(self.upper_y(z)) + lift, z)
                             for z in self.z_column + self.pitch * np.arange(1, self.eggs // 2)]
        return lower + upper

    def egg_path(self, egg=LARGE, step=2.0):
        """Every centre an egg passes through on its way, (y, z): for swept clearance."""
        lift = centre_offset(egg) - self.c_off
        path = []
        for z in np.arange(self.z_front + 6, self.z_column, -step):  # Lower, incl. cradle
            path.append((float(self.lower_y(min(z, self.z_front))) + lift, z))
        y_lo = float(self.lower_y(self.z_column)) + lift
        y_top = float(self.upper_y(self.z_column)) + lift
        for y in np.arange(y_lo, y_top, step):                         # down the column
            path.append((y, self.z_column))
        for z in np.arange(self.z_column, self.z_front + 6, step):    # Upper, incl. lip
            path.append((float(self.upper_y(min(z, self.z_front))) + lift, z))
        return np.array(path)


if __name__ == "__main__":
    L = Layout()
    print(f"length {L.length} mm, height {L.height:.1f} mm, width {2 * (SHELL_X + STRUT_R):.1f} mm")
    print(f"egg centre above Rails: Large {L.c_off:.2f}, XL {L.c_off_xl:.2f}")
    print(f"column z {L.z_column:.1f}, front eggs z {L.z_front:.1f}")
    print(f"Lower centre y: front {float(L.lower_y(L.z_front)):.1f}, back {float(L.lower_y(L.z_column)):.1f}")
    print(f"Upper centre y: back {float(L.upper_y(L.z_column)):.1f}, front {float(L.upper_y(L.z_front)):.1f}")
    print(f"eggs placed: {len(L.egg_centres())}")
