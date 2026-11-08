"""Turn a build/ mesh from the print frame (z = length) into the use frame for viewing:
length along X, across along Y, up along Z (the viewer's up). Writes <name>_use.stl."""

import sys
from pathlib import Path

import numpy as np
import trimesh

for name in sys.argv[1:]:
    m = trimesh.load(name, force="mesh")
    x, y, z = m.vertices.T.copy()
    m.vertices = np.column_stack([z, -x, y])     # a proper rotation: no mirror
    out = Path(name).with_name(Path(name).stem + "_use.stl")
    m.export(out)
    print(out)
