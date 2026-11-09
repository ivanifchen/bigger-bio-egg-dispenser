# Bigger Bio Egg Dispenser

A 16-egg remix of pymata's Bio Egg Dispenser, sized for a 16 in fridge shelf: a 389 mm lattice
cage that sits front-to-back on the shelf. Load eggs at the top front; they roll back, turn
down, and roll forward to where you take them out.

- Page: https://ivanifchen.github.io/bigger-bio-egg-dispenser/
- Fit coupon: [exports/egg_dispenser_fit_coupon_plate.3mf](exports/egg_dispenser_fit_coupon_plate.3mf)
- Back half: [exports/egg_dispenser_back_plate.3mf](exports/egg_dispenser_back_plate.3mf)
- Front half: [exports/egg_dispenser_front_plate.3mf](exports/egg_dispenser_front_plate.3mf)

Print the fit coupon first and pick the pin fit that grips. Both halves print standing, each
about 195 mm tall, in Bambu PLA Matte at 0.20 mm layers with 3 walls, 10% gyroid and an 8 mm
brim; no supports. About 17 h and 285 g for both on an X2D.

There is no STEP export: the lattice is generated as a mesh (see `docs/adr/0004-mesh-pipeline-not-build123d.md`).

## Source

The geometry is a Python mesh pipeline (numpy, trimesh, manifold3d, scikit-image):

- `src/layout.py`: the shared egg path, rails and side guides
- `src/grown.py`: the generated lattice cage
- `src/joint.py`, `src/halves.py`: the seam, pins and print jobs
- `src/plate.py`: Bambu Studio project writer
- `src/check.py`: print rules, egg clearance and seam measurements
- `src/exports.py`: rebuilds the plate 3MFs in `exports/`
- `docs/adr/`: design decisions
